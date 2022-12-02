def getInfix(exp) :
    s = []
    exp = exp.split()
    for i in exp:    
        if (isNumber(i)) :        
            s.insert(0, i)
        else:
            op1 = s[0]
            s.pop(0)
            op2 = s[0]
            s.pop(0)
            s.insert(0,"("+op2+i+op1+")")
    return s[0]


def eval(str):
    assert(len(str.split())==3)
    inp1, opr, inp2 = str.split()
    if opr =="+":
        return float(inp1) + float(inp2)
    elif opr =="-":
        return float(inp1) - float(inp2)
    elif opr =="*":
        return float(inp1) * float(inp2)
    elif opr =="/":
        return float(inp1) / float(inp2)
    return None


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def evaluatePostfix(exp):
    exp = exp.split()
    stack = []
    for i in exp:
        if isNumber(i):
            stack.append(i)
        else:
            val1 = stack.pop()
            val2 = stack.pop()
            stack.append(str(eval(val2 +' '+ i +' '+ val1)))

    return float(stack.pop())

print(evaluatePostfix("100 200 + "))