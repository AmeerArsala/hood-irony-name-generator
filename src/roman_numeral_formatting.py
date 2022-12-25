roman_numeral_values = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000
}


# @return result: str
def roman_to_integer(s: str):
    # Initialize the result to 0
    result = 0

    # Iterate through the string, adding the value of each Roman numeral
    # to the result
    for i in range(len(s)):
        # If the current numeral is greater than or equal to the next numeral,
        # add its value to the result
        if i == len(s) - 1 or roman_numeral_values[s[i]] >= roman_numeral_values[s[i+1]]:
            result += roman_numeral_values[s[i]]
        # If the current numeral is less than the next numeral, subtract its value
        # from the result
        else:
            result -= roman_numeral_values[s[i]]

    return str(result)


# @return roman_num: str
def integer_to_roman(number: str):
    num = int(number)

    val = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    syb = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
    roman_num = ""

    for i in range(len(val)):
        count = int(num / val[i])
        roman_num += syb[i] * count
        num -= val[i] * count

    return roman_num

