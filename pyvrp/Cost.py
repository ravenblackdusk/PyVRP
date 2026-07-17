import numbers


class Cost:
    """
    Compound cost value with two components: the number of missing
    soft-required clients and the monetary cost.

    Comparisons between two :class:`Cost` objects are lexicographic:
    ``missing_soft_required`` first, then ``cost``.
    """

    __slots__ = ("_msr", "_cost")

    def __init__(self, msr_or_cost, cost=None):
        if cost is None:
            self._msr = 0
            self._cost = int(msr_or_cost)
        else:
            self._msr = int(msr_or_cost)
            self._cost = int(cost)

    @property
    def missing_soft_required(self) -> int:
        return self._msr

    @property
    def cost(self) -> int:
        return self._cost

    # --- numeric protocol ---------------------------------------------------

    def __int__(self):
        return self._cost

    def __float__(self):
        return float(self._cost)

    def __index__(self):
        return self._cost

    def __bool__(self):
        return self._msr != 0 or self._cost != 0

    def __round__(self, ndigits=None):
        return round(self._cost, ndigits)

    def __abs__(self):
        return abs(self._cost)

    def __neg__(self):
        return -self._cost

    # --- comparison (lexicographic, Cost-vs-Cost only) -----------------------

    @staticmethod
    def _key(other):
        """
        Comparison key: plain numbers are treated as a Cost with zero missing
        soft-required clients.
        """
        if isinstance(other, Cost):
            return (other._msr, other._cost)
        if isinstance(other, numbers.Real):
            return (0, other)
        return None

    def __eq__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) == key

    def __ne__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) != key

    def __lt__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) < key

    def __le__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) <= key

    def __gt__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) > key

    def __ge__(self, other):
        key = Cost._key(other)
        if key is None:
            return NotImplemented
        return (self._msr, self._cost) >= key

    # --- arithmetic (delegates to cost component, returns plain numbers) -----

    def __add__(self, other):
        if isinstance(other, Cost):
            return self._cost + other._cost
        return self._cost + other

    def __radd__(self, other):
        return other + self._cost

    def __sub__(self, other):
        if isinstance(other, Cost):
            return self._cost - other._cost
        return self._cost - other

    def __rsub__(self, other):
        return other - self._cost

    def __mul__(self, other):
        return self._cost * other

    def __rmul__(self, other):
        return other * self._cost

    def __truediv__(self, other):
        if isinstance(other, Cost):
            return self._cost / other._cost
        return self._cost / other

    def __rtruediv__(self, other):
        return other / self._cost

    def __floordiv__(self, other):
        if isinstance(other, Cost):
            return self._cost // other._cost
        return self._cost // other

    def __rfloordiv__(self, other):
        return other // self._cost

    # --- misc ----------------------------------------------------------------

    def __hash__(self):
        # A Cost without missing soft-required clients compares equal to its
        # plain cost value, so it must also hash like it.
        if self._msr == 0:
            return hash(self._cost)
        return hash((self._msr, self._cost))

    def __reduce__(self):
        return (Cost, (self._msr, self._cost))

    def __str__(self):
        return str(self._cost)

    def __repr__(self):
        if self._msr:
            return f"Cost({self._msr}, {self._cost})"
        return f"Cost({self._cost})"


numbers.Number.register(Cost)
