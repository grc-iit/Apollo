import json,os
def bashing(command,printToScreen=True):
    os.system(command + " > QbashFile")
    with open("QbashFile") as f:
        var=f.read()
    os.remove('QbashFile')
    if printToScreen:
        print(var)
    else:
        return var
def get_bandwidth(ordering='sequential'):
    if ordering == 'sequential':
        output = bashing('ior -i=5',False)
    else:
        output = bashing('ior -z -i=5',False)
    if not output:
        print("It seems like ior is not installed in this system")
    relevantRows=['write','read']
    for line in output[output.find("ordering in a file"):].split('\n'): 
        #Getting type of access, sequential vs random 
        ordering = line.split(':')
        ordering = ordering[1][1:]
        sequential={ordering : {'write':[],'read':[]}}
        c=0
        s=0
        break
    for line in output[output.find("Results"):output.find("Summary")].split('\n'):
        parsing = line.split()
        if parsing and parsing[0] in relevantRows:
            c+=1
            sequential[ordering][parsing[0]].append(float(parsing[1]))
    c=c/len(relevantRows)
    for row in relevantRows:
        s = sum(sequential[ordering][row])
        m = min(sequential[ordering][row])
        M = max(sequential[ordering][row])
        sequential[ordering][row]={'min':m,'max':M,'avg':s/c}
    return sequential
data = [get_bandwidth(ordering='sequential'),get_bandwidth(ordering='random')]
with open('bandwidth.json','w') as f:
    json.dump(obj=data,fp=f)