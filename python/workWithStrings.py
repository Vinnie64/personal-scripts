# This is another project working with strings
# Using functions as well

phrase = "Giraffe Academy"

# .lower() will print to lower
print(phrase.lower() + " is something I have never heard of")

# .upper() will make everything upper case
print(phrase.upper() + " is something I have never heard of")

# .isupper and .islower will test if the ENTIRE string is upper or lower case

print(phrase.upper().isupper())
print(phrase.lower().islower())

# to get the length of a string use len(str)
print(len(phrase))

# you can also access certain letters of a string
# to do this you would do str[index]
# index starts at 0
print("The first letter of the phrase is: " + phrase[0])

# you can also use .index("val") to find where a specific character is located
# in any given string, it returns the FIRST index of it
print(phrase.index("a"))

# you can also use .replace("","") the first value is what you WANT to replace
# the second value is what will be put in place of the first
print(phrase.replace("Giraffe", "Dog"))
