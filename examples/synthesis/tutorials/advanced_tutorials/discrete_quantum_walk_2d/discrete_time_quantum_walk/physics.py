"""The frozen walk starts at (16,16) with a signed four-direction coin."""

POSITION_AMPLITUDES = tuple(1.0 if word == 16 else 0.0 for word in range(32))
COIN_AMPLITUDES = (0.5, 0.5, -0.5, -0.5)
