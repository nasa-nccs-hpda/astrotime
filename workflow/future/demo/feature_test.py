from decimal import getcontext, Decimal
from typing import List, Optional, Dict, Type, Union, Tuple, Any

def float_to_binary_2(fval: float, places: int =64) -> List[int]:
	fractional_binary = bin(int(fval*pow(2,places)))[2:]
	return [int(bit) for bit in fractional_binary]

def float_to_binary_1(fval: float, places: int =64) -> List[int]:
	fractional_binary: str = ""
	for _ in range(places):
		fval *= 2
		bit = int(fval)
		print( f" fval={fval:.4f}, bit={bit}")
		fractional_binary += str(bit)
		fval -= bit
	return [int(bit) for bit in fractional_binary]

def float_to_binary_0(num, places=64) -> List[int]:
	getcontext().prec = places
	decimal_num = Decimal(str(num))
	integer_part = int(decimal_num)
	fractional_part = decimal_num - integer_part

	integer_binary = bin(integer_part)[2:]  # remove '0b' prefix

	fractional_binary = ""
	for _ in range(places):
		fractional_part *= 2
		bit = int(fractional_part)
		fractional_binary += str(bit)
		fractional_part -= bit

	return [int(bit) for bit in fractional_binary]

if __name__ == "__main__":
	fval = 0.234

	print( float_to_binary_0(fval, places=64) )
	print( float_to_binary_1(fval, places=64) )
	print( float_to_binary_2(fval, places=64) )