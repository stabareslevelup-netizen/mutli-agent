"""
engine/tools/social_apis.py — platform adapters (X, Instagram, YouTube).

Each adapter encodes the REAL platform constraints as its own rules (not
hardcoded around). All three are built; only X is `live` in v1. Publishing is
gated: an adapter only publishes when it is live AND a human confirm is passed
(see distribution.py). HTTP transports are injected so nothing reaches the
network in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class StagedPost:
    platform: str
    payload: dict
    contains_url: bool = False
    estimated_cost_usd: float = 0.0
    cost_breakdown: list[dict] = field(default_factory=list)   # per-post cost lines
    notes: str = ""


class PublishBlocked(Exception):
    """Adapter is not live, or publish attempted without human confirm."""


@runtime_checkable
class PlatformAdapter(Protocol):
    name: str
    live: bool
    requires_setup: Optional[str]
    def stage(self, content: dict) -> StagedPost: ...
    async def publish(self, staged: StagedPost, *, http: Any, confirmed: bool) -> dict: ...


def _has_url(text: str) -> bool:
    return "http://" in text or "https://" in text


class XAdapter:
    """X API v2 — pay-per-use. A post containing a URL costs ~$0.20 vs ~$0.01
    without (real X pricing). `link_mode` is configurable:
      - 'inline': link in the main post (one URL post)
      - 'reply' : clean main post (cheap, better reach) + link in a reply post
      - 'none'  : no link
    Per-post cost is itemized so distribution can log it via cost_guard."""
    name = "x"
    live = True
    requires_setup = None

    def __init__(self, link_mode: str = "reply", cost_with_url: float = 0.20,
                 cost_without_url: float = 0.01):
        self.link_mode = link_mode
        self.cost_with_url = cost_with_url
        self.cost_without_url = cost_without_url

    def stage(self, content: dict) -> StagedPost:
        thread: list[str] = list(content.get("x_thread", []))
        link: Optional[str] = content.get("link")
        posts: list[dict] = [{"text": t} for t in thread] or [{"text": content.get("text", "")}]
        breakdown: list[dict] = []

        if link and self.link_mode == "inline":
            posts[0]["text"] = f'{posts[0]["text"]} {link}'.strip()
        for p in posts:
            url = _has_url(p["text"])
            cost = self.cost_with_url if url else self.cost_without_url
            breakdown.append({"text": p["text"][:40], "url": url, "usd": cost})
        if link and self.link_mode == "reply":
            posts.append({"text": link, "reply_to_prev": True})
            breakdown.append({"text": link[:40], "url": True, "usd": self.cost_with_url})

        total = round(sum(b["usd"] for b in breakdown), 4)
        contains_url = any(b["url"] for b in breakdown)
        return StagedPost(platform=self.name, payload={"posts": posts, "link_mode": self.link_mode},
                          contains_url=contains_url, estimated_cost_usd=total,
                          cost_breakdown=breakdown,
                          notes=f"{len(posts)} post(s), link_mode={self.link_mode}")

    async def publish(self, staged: StagedPost, *, http: Any, confirmed: bool) -> dict:
        if not confirmed:
            raise PublishBlocked("X publish requires explicit human confirm")
        # http is the injected X client; in v1 this runs only after confirm.
        return await http.post_thread(staged.payload["posts"])


class InstagramAdapter:
    """Instagram Graph API. Requires Business account + linked FB Page +
    approved instagram_business_content_publish. Media must already be at a
    PUBLIC URL. Two-step: create container -> publish, with status polling.
    Cap: 25 posts / 24h. Not live until Meta app review completes."""
    name = "instagram"
    live = False
    requires_setup = ("requires Meta app review (2-4 wk): Business account + linked FB Page "
                      "+ approved instagram_business_content_publish")
    DAILY_CAP = 25

    def stage(self, content: dict) -> StagedPost:
        media_url = content.get("media_url")
        if not media_url:
            raise PublishBlocked("Instagram requires media at a public URL before publish")
        return StagedPost(platform=self.name,
                          payload={"caption": content.get("ig_caption", ""), "media_url": media_url,
                                   "steps": ["create_container", "poll_status", "publish"]},
                          notes=f"two-step publish; daily cap {self.DAILY_CAP}/24h")

    async def publish(self, staged: StagedPost, *, http: Any, confirmed: bool) -> dict:
        if not self.live:
            raise PublishBlocked(self.requires_setup)
        if not confirmed:
            raise PublishBlocked("Instagram publish requires explicit human confirm")
        container = await http.create_container(staged.payload["media_url"], staged.payload["caption"])
        await http.poll_container(container["id"])
        return await http.publish_container(container["id"])


class YouTubeAdapter:
    """YouTube Data API v3. OAuth + resumable upload; quota-limited. Not live
    until OAuth is configured. YouTube cannot publish without a video file —
    a text-only brand naturally has no asset, so this adapter stages nothing
    and is excluded from the bundle (same pattern as Instagram's media
    requirement)."""
    name = "youtube"
    live = False
    requires_setup = "requires OAuth + YouTube Data API v3 setup (quota-limited)"

    def stage(self, content: dict) -> StagedPost:
        video_url = content.get("asset_url")
        if not video_url:
            raise PublishBlocked("YouTube requires a video file to upload")
        return StagedPost(platform=self.name,
                          payload={"title": content.get("title", "")[:100],
                                   "description": content.get("youtube_script", ""),
                                   "video_url": video_url,
                                   "privacyStatus": "private"},
                          notes="OAuth + Data API v3 resumable upload; quota-limited")

    async def publish(self, staged: StagedPost, *, http: Any, confirmed: bool) -> dict:
        if not self.live:
            raise PublishBlocked(self.requires_setup)
        if not confirmed:
            raise PublishBlocked("YouTube publish requires explicit human confirm")
        return await http.upload(staged.payload)
