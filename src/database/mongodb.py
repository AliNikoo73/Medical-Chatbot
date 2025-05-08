"""MongoDB database module for the medical chatbot."""
from typing import Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

class MongoDBManager:
    """Manager class for MongoDB operations."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017/",
        db_name: str = "clinic_chatbot",
        collection_name: str = "conversations",
    ) -> None:
        """Initialize the MongoDB manager.

        Args:
            uri: MongoDB connection URI.
            db_name: Name of the database.
            collection_name: Name of the collection.
        """
        self.client: MongoClient = MongoClient(uri)
        self.db: Database = self.client[db_name]
        self.collection: Collection = self.db[collection_name]

    def save_conversation(
        self, symptom: str, role: str, message: str, metadata: Optional[Dict] = None
    ) -> None:
        """Save a conversation entry to the database.

        Args:
            symptom: The symptom being discussed.
            role: The role of the message sender (user/bot).
            message: The message content.
            metadata: Optional metadata to store with the conversation.
        """
        document = {
            "symptom": symptom,
            "role": role,
            "message": message,
            "timestamp": self.client.server_time(),
        }
        if metadata:
            document.update(metadata)
        self.collection.insert_one(document)

    def get_conversations(
        self, symptom: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Retrieve conversations from the database.

        Args:
            symptom: Optional symptom to filter by.
            limit: Maximum number of conversations to retrieve.

        Returns:
            List of conversation documents.
        """
        query = {"symptom": symptom} if symptom else {}
        return list(self.collection.find(query).sort("timestamp", -1).limit(limit))

    def clear_conversations(self) -> None:
        """Clear all conversations from the database."""
        self.collection.delete_many({})

    def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close() 