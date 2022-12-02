from matplotlib import pyplot as plt
import ast

with open("result/bleu.txt",'r') as f:
    l = ast.literal_eval(f.readlines()[0])

print(float(sum(l))/len(l))

with open("result/test_loss.txt",'r') as f:
    l1 = ast.literal_eval(f.readlines()[0])

with open("result/train_loss.txt",'r') as f:
    l2 = ast.literal_eval(f.readlines()[0])



plt.plot(range(1,301),l, label="bleu score")
plt.xlabel("epochs")
plt.ylabel("BLEU score")
plt.grid()
plt.legend()
plt.title("BLEU score vs epochs (Infix)")
plt.savefig("plots/bleu.png")

plt.clf()

plt.plot(range(1,301),l1, label="bleu score")
plt.xlabel("epochs")
plt.ylabel("Training loss")
plt.grid()
plt.legend()
plt.title("Train loss vs epochs (Infix)")
plt.savefig("plots/train_loss.png")

plt.clf()

plt.plot(range(1,301),l2, label="bleu score")
plt.xlabel("epochs")
plt.ylabel("Val loss")
plt.grid()
plt.legend()
plt.title("Validation loss vs epochs (Infix)")
plt.savefig("plots/valid_loss.png")
