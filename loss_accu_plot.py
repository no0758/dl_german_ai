

file  = open('logger_resnet_glr.log','r')

loss=[]
accu=[]

for w in file:
    if 'evaluate_loss' in w:
        ss = w.split(',')
        for s in ss:
            if 'evaluate_loss' in s:
                loss.append(s.split(':')[1])
            elif 'accu' in s:
                accu.append(s.split(':')[1].replace('.\n', ''))
