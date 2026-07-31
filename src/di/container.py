"""Minimal, explicit dependency-injection container.

No autowiring, no magic — just a registry that maps interface types
to factory callables or singleton instances.
"""

from __future__ import annotations

from typing import Any, Callable


class DIContainer:
    """A simple DI container that maps types to factories.

    Usage::

        container = DIContainer()
        container.register(LLMClient, lambda c: ClaudeClient(api_key="..."))
        client = container.resolve(LLMClient)   # same instance every time (singleton)
    """

    def __init__(self) -> None:
        self._factories: dict[Any, Callable[[DIContainer], Any]] = {}
        self._singletons: dict[Any, Any] = {}
        self._is_singleton: dict[Any, bool] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        interface: type,
        factory: Callable[[DIContainer], Any],
        singleton: bool = True,
    ) -> None:
        """Register *factory* as the producer for *interface*.

        When *singleton* is ``True`` (the default) the factory is called
        once and the same instance is returned on every ``resolve`` call.
        """
        self._factories[interface] = factory
        self._is_singleton[interface] = singleton
        # clear any previously cached singleton
        self._singletons.pop(interface, None)

    # ------------------------------------------------------------------
    def resolve(self, interface: type) -> Any:
        """Return an instance for *interface*.

        If registered as a singleton the same instance is returned on
        every call.  Raises ``KeyError`` if *interface* was never
        registered.
        """
        if interface in self._singletons:
            return self._singletons[interface]

        factory = self._factories.get(interface)
        if factory is None:
            msg = f"No factory registered for {interface.__name__}"
            raise KeyError(msg)

        instance = factory(self)

        if self._is_singleton.get(interface, True):
            self._singletons[interface] = instance

        return instance
