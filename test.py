# Exec changing caller function namespace
def exec_code(code):
    local_vars = {}
    exec(code,{},local_vars)
    
    return local_vars

code = '''
x = 10
y = 20
a = 2.7
z = x + y + a
'''
print(exec_code(code))