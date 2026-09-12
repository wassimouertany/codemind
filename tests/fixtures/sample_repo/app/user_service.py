"""Mirror of the Java bug in Python, for cross-language chunking tests."""

from app.user_repository import UserRepository


class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user_by_id(self, user_id: int) -> dict:
        user = self._repository.find_by_id(user_id)
        if user is None:
            raise KeyError(user_id)
        return user

    def deactivate_user(self, user_id: int) -> None:
        user = self.get_user_by_id(user_id)
        user["active"] = False
        self._repository.save(user)
