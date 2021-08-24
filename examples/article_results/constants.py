import matplotlib.pyplot as plt
import string

def cm2inch(value):
    return value/2.54

#figsizes
f26_size = (cm2inch(12.3), cm2inch(10))
f45_size = (cm2inch(12.3), cm2inch(18))
fbig_size = (cm2inch(17.3), cm2inch(12))
save_kwargs = {'dpi': 300}

# Tagging properties
tagging = string.ascii_uppercase
# tagging = string.ascii_lowercase
tag_kwargs = {'size':14, 'weight':'bold'}

basic_colors =  ['tab:blue', 'tab:orange', 'tab:green', 'r', 'purple', 'b']
