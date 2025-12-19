from dataclasses import dataclass


@dataclass
class User:
user_id: int
age_bucket: str
country: str


@dataclass
class Item:
item_id: int
category: str
creator_id: int


@dataclass
class Interaction:
user_id: int
item_id: int
event: str # impression / click / watch
timestamp: int