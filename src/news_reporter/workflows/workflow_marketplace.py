"""Workflow Marketplace - Sharing, discovery, and marketplace features"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.graph_schema import GraphDefinition
from .workflow_persistence import WorkflowRecord

logger = logging.getLogger(__name__)


class MarketplaceCategory(str, Enum):
    """Marketplace categories"""
    DATA_PROCESSING = "data_processing"
    AI_ML = "ai_ml"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    ANALYTICS = "analytics"
    CUSTOM = "custom"


class ListingStatus(str, Enum):
    """Listing status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REMOVED = "removed"


@dataclass
class MarketplaceListing:
    """A workflow listing in the marketplace"""
    listing_id: str
    workflow_id: str
    title: str
    description: str
    category: MarketplaceCategory
    author_id: str
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    price: float = 0.0  # 0 = free
    currency: str = "USD"
    status: ListingStatus = ListingStatus.DRAFT
    download_count: int = 0
    rating: float = 0.0
    review_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    preview_image: Optional[str] = None
    documentation_url: Optional[str] = None


@dataclass
class MarketplaceReview:
    """A review for a marketplace listing"""
    review_id: str
    listing_id: str
    user_id: str
    rating: int  # 1-5
    comment: str
    created_at: Optional[datetime] = None
    helpful_count: int = 0


class WorkflowMarketplace:
    """Manages workflow marketplace, sharing, and discovery"""
    
    def __init__(self):
        self.listings: Dict[str, MarketplaceListing] = {}
        self.reviews: Dict[str, List[MarketplaceReview]] = {}  # listing_id -> [reviews]
        self._listing_counter = 0
        self._review_counter = 0
    
    def create_listing(
        self,
        listing_id: str,
        workflow_id: str,
        title: str,
        description: str,
        category: MarketplaceCategory,
        author_id: str,
        tags: Optional[List[str]] = None,
        price: float = 0.0
    ) -> MarketplaceListing:
        """Create a marketplace listing"""
        listing = MarketplaceListing(
            listing_id=listing_id,
            workflow_id=workflow_id,
            title=title,
            description=description,
            category=category,
            author_id=author_id,
            tags=tags or [],
            price=price,
            created_at=datetime.now()
        )
        
        self.listings[listing_id] = listing
        logger.info(f"Created marketplace listing: {listing_id}")
        return listing
    
    def publish_listing(self, listing_id: str) -> bool:
        """Publish a listing"""
        listing = self.listings.get(listing_id)
        if not listing:
            return False
        
        listing.status = ListingStatus.PUBLISHED
        listing.updated_at = datetime.now()
        logger.info(f"Published listing: {listing_id}")
        return True
    
    def search_listings(
        self,
        query: Optional[str] = None,
        category: Optional[MarketplaceCategory] = None,
        tags: Optional[List[str]] = None,
        min_rating: float = 0.0,
        max_price: Optional[float] = None,
        limit: int = 20
    ) -> List[MarketplaceListing]:
        """Search marketplace listings"""
        results = [
            l for l in self.listings.values()
            if l.status == ListingStatus.PUBLISHED
        ]
        
        # Filter by category
        if category:
            results = [l for l in results if l.category == category]
        
        # Filter by tags
        if tags:
            results = [
                l for l in results
                if any(tag in l.tags for tag in tags)
            ]
        
        # Filter by rating
        results = [l for l in results if l.rating >= min_rating]
        
        # Filter by price
        if max_price is not None:
            results = [l for l in results if l.price <= max_price]
        
        # Text search
        if query:
            query_lower = query.lower()
            results = [
                l for l in results
                if query_lower in l.title.lower() or
                   query_lower in l.description.lower() or
                   any(query_lower in tag.lower() for tag in l.tags)
            ]
        
        # Sort by rating and download count
        results.sort(key=lambda l: (l.rating, l.download_count), reverse=True)
        
        return results[:limit]
    
    def get_listing(self, listing_id: str) -> Optional[MarketplaceListing]:
        """Get a listing by ID"""
        return self.listings.get(listing_id)
    
    def add_review(
        self,
        listing_id: str,
        user_id: str,
        rating: int,
        comment: str
    ) -> MarketplaceReview:
        """Add a review to a listing"""
        review = MarketplaceReview(
            review_id=f"review_{self._review_counter}",
            listing_id=listing_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            created_at=datetime.now()
        )
        
        self._review_counter += 1
        
        if listing_id not in self.reviews:
            self.reviews[listing_id] = []
        self.reviews[listing_id].append(review)
        
        # Update listing rating
        listing = self.listings.get(listing_id)
        if listing:
            all_reviews = self.reviews[listing_id]
            listing.rating = sum(r.rating for r in all_reviews) / len(all_reviews) if all_reviews else 0.0
            listing.review_count = len(all_reviews)
        
        logger.info(f"Added review for listing {listing_id}")
        return review
    
    def get_reviews(self, listing_id: str, limit: int = 10) -> List[MarketplaceReview]:
        """Get reviews for a listing"""
        reviews = self.reviews.get(listing_id, [])
        reviews.sort(key=lambda r: r.created_at or datetime.min, reverse=True)
        return reviews[:limit]
    
    def download_workflow(self, listing_id: str) -> Optional[str]:
        """Download a workflow from marketplace (returns workflow_id)"""
        listing = self.listings.get(listing_id)
        if not listing or listing.status != ListingStatus.PUBLISHED:
            return None
        
        listing.download_count += 1
        listing.updated_at = datetime.now()
        logger.info(f"Downloaded workflow from listing {listing_id}")
        return listing.workflow_id
    
    def get_popular_listings(self, limit: int = 10) -> List[MarketplaceListing]:
        """Get most popular listings"""
        listings = [
            l for l in self.listings.values()
            if l.status == ListingStatus.PUBLISHED
        ]
        listings.sort(key=lambda l: (l.download_count, l.rating), reverse=True)
        return listings[:limit]
    
    def get_recent_listings(self, limit: int = 10) -> List[MarketplaceListing]:
        """Get recently published listings"""
        listings = [
            l for l in self.listings.values()
            if l.status == ListingStatus.PUBLISHED
        ]
        listings.sort(key=lambda l: l.created_at or datetime.min, reverse=True)
        return listings[:limit]


# Global marketplace instance
_global_marketplace = WorkflowMarketplace()


def get_workflow_marketplace() -> WorkflowMarketplace:
    """Get the global workflow marketplace instance"""
    return _global_marketplace
