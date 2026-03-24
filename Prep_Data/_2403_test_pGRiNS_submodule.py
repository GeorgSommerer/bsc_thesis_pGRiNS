
import sys
print(sys.path)

sys.path.append('.')
print(sys.path)

from grins import reg_funcs

print(reg_funcs.psH(5,-3,0.3,2))