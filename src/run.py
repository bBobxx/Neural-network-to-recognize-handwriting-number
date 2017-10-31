import mnist_loader
import network
from pylab import *
test_re,test_inputs,training_data, validation_data, test_data= mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
print("please enter the number of the picture you want to recognize(from 0 to 9999):")
m=int(raw_input())
print ("the picture you want to recognize is shown in the image.")
a=test_inputs[m].reshape((28,28))
figure()
imshow(a)
show()
print("here is the result")
re=np.argmax(net.feedforward(test_inputs[m]))
print(re)
#)

#print (test_re[0])
#a=test_inputs[0].reshape((28,28))
#figure()
#imshow(a)
#show()
#hold"""