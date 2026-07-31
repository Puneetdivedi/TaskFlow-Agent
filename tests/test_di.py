"""Tests for the dependency-injection container and factory."""

from __future__ import annotations

from typing import Protocol

import pytest

from src.di.container import DIContainer


# ---------------------------------------------------------------------------
# DIContainer
# ---------------------------------------------------------------------------
class _Animal(Protocol):
    def speak(self) -> str: ...


class _Dog:
    def speak(self) -> str:
        return "woof"


class _Cat:
    def speak(self) -> str:
        return "meow"


class TestDIContainer:
    def test_register_and_resolve(self) -> None:
        container = DIContainer()
        container.register(_Animal, lambda c: _Dog())
        animal = container.resolve(_Animal)
        assert isinstance(animal, _Dog)
        assert animal.speak() == "woof"

    def test_singleton_returns_same_instance(self) -> None:
        container = DIContainer()
        container.register(_Animal, lambda c: _Dog())
        a1 = container.resolve(_Animal)
        a2 = container.resolve(_Animal)
        assert a1 is a2

    def test_non_singleton_returns_new_instance(self) -> None:
        container = DIContainer()
        container.register(_Animal, lambda c: _Dog(), singleton=False)
        a1 = container.resolve(_Animal)
        a2 = container.resolve(_Animal)
        assert a1 is not a2

    def test_re_registration_clears_singleton_cache(self) -> None:
        container = DIContainer()
        container.register(_Animal, lambda c: _Dog())
        dog = container.resolve(_Animal)
        container.register(_Animal, lambda c: _Cat())
        cat = container.resolve(_Animal)
        assert isinstance(cat, _Cat)
        assert dog is not cat

    def test_resolve_unregistered_raises_key_error(self) -> None:
        container = DIContainer()
        with pytest.raises(KeyError, match="No factory registered for _Animal"):
            container.resolve(_Animal)
