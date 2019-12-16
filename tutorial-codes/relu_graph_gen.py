import torch
from matplotlib import pyplot as plt

def ReLU_wrapper(i):
    relu = torch.nn.ReLU()
    int_to_tensor = torch.Tensor([i])
    return relu(int_to_tensor).data

if __name__ == "__main__":

    print("CSOAI - Generating Relu graph for you")

    input_range = []
    relu_output = [] 

    # generate data points
    for i in range(-5,5,1):
        input_range.append(i)
        relu_output.append(ReLU_wrapper(i))

    # plot using matplotlib
    plt.plot(input_range, relu_output)
    plt.ylabel("RELU")
    plt.show()