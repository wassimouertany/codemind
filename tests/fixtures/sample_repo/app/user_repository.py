class UserRepository:
    def __init__(self, db):
        self._db = db

    def find_by_id(self, user_id: int):
        return self._db.get(user_id)

    def save(self, user: dict) -> None:
        self._db[user["id"]] = user
