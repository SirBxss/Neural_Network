import numpy as np
import sklearn
from Optimization.Loss import CrossEntropyLoss


def main():
    # batch_size = 9
    # num_classes = 4
    #
    # # Create a label tensor: one-hot encoded with all labels set to class 2
    # label_tensor = np.zeros((batch_size, num_classes))
    # label_tensor[:, 2] = 1
    #
    # print(f"Label Tensor : {label_tensor}")
    #
    # # Create an input tensor: all predictions set to class 1
    # prediction_tensor = np.zeros_like(label_tensor)
    # prediction_tensor[:, 1] = 1
    #
    # print(f"Input : {prediction_tensor}")
    #
    # # Instantiate the CrossEntropyLoss class
    # loss_layer = CrossEntropyLoss()
    #
    # # Calculate loss
    # loss = loss_layer.forward(prediction_tensor, label_tensor)
    #
    # # Expected loss value (calculated manually or from a known correct source)
    # expected_loss = 324.3928805  # Provided expected loss
    #
    # # Print the calculated loss
    # print(f"Calculated loss: {loss}")
    #
    # # Check if the calculated loss matches the expected loss within 4 decimal places
    # try:
    #     np.testing.assert_almost_equal(loss, expected_loss, decimal=4)
    #     print("Test passed!")
    # except AssertionError as e:
    #     print(f"Test failed! {e}")

    # prediction_tensor = np.array([[0.8, 0.1, 0.1],
    #                               [0.2, 0.7, 0.1],
    #                               [0.0, 0.3, 0.6]])
    # label_tensor = np.array([[0, 1, 0],
    #                          [1, 0, 0],
    #                          [0, 0, 1]])

    # label_tensor = np.zeros((9, 4))
    # label_tensor[:, 2] = 1

    # ---------------------------------------------------------------------------------------------

    # label_tensor = np.array([[0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 1., 0.]])
    # ---------------------------------------------------------------------------------------------

    # label_tensor = np.array([[0., 0., 1., 0.],
    #                          [0., 0., 1., 0.],
    #                          [0., 0., 1., 0.],
    #                          [0., 0., 1., 0.]])
    #
    # print(f"Label tensor \n {label_tensor}")

    # prediction_tensor = np.zeros_like(label_tensor)
    # prediction_tensor[:, 1] = 1

    # prediction_tensor = np.array([[0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 1., 0., 0.]])
    # ---------------------------------------------------------------------------------------------
    # prediction_tensor = np.array([[0., 1., 0., 0.],
    #                               [0., 1., 0., 0.],
    #                               [0., 1., 0., 0.],
    #                               [0., 1., 0., 0.]])
    #
    # print(f"Prediction tensor \n {prediction_tensor}")
    #


    batch_size = 9
    categories = 4
    label_tensor = np.zeros((batch_size, categories))
    label_tensor[:, 2] = 1
    prediction_tensor = np.zeros_like(label_tensor)
    prediction_tensor[:, 1] = 1

    # Instantiate the CrossEntropyLoss class
    cross_entropy_loss = CrossEntropyLoss()

    # Call the forward method
    loss = cross_entropy_loss.forward(prediction_tensor, label_tensor)
    print(f"The Loss is: \n {loss}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print(sklearn.__version__)
    main()
