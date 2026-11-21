#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_HIDDEN_NEURONS 16
#define NUM_OUTPUT_NEURONS 10
#define LEARNING_RATE 0.01f

typedef struct
{
    uint8_t *data;
    uint32_t image_count;
    uint32_t rows;
    uint32_t cols;
} Images;

void forward_pass(const Images *images, size_t image_idx, float *weights1, float *weights2, float *activation1, float *output, float *bias1, float *bias2, float *probabilities)
{
    // Set neurons to 0 before next forward pass
    memset(activation1, 0, NUM_HIDDEN_NEURONS * sizeof(*activation1));
    memset(output, 0, NUM_OUTPUT_NEURONS * sizeof(*output));

    size_t pixels_per_image = images->rows * images->cols;
    size_t image_start = image_idx * pixels_per_image;

    // First hidden layer
    for (size_t neuron_idx = 0; neuron_idx < NUM_HIDDEN_NEURONS; neuron_idx++)
    {
        for (size_t pixel_idx = 0; pixel_idx < pixels_per_image; pixel_idx++)
        {
            // Normalizing pixels from 0-255 to 0-1
            float normalized = images->data[image_start + pixel_idx] / 255.0f;
            activation1[neuron_idx] += weights1[neuron_idx * pixels_per_image + pixel_idx] * normalized;
        }

        activation1[neuron_idx] += bias1[neuron_idx];

        // ReLU
        if (activation1[neuron_idx] < 0)
        {
            activation1[neuron_idx] = 0;
        }
    }

    // Output layer
    for (size_t output_idx = 0; output_idx < NUM_OUTPUT_NEURONS; output_idx++)
    {
        for (size_t hidden_idx = 0; hidden_idx < NUM_HIDDEN_NEURONS; hidden_idx++)
        {
            output[output_idx] += activation1[hidden_idx] * weights2[output_idx * NUM_HIDDEN_NEURONS + hidden_idx];
        }

        output[output_idx] += bias2[output_idx];
    }

    // Softmax portion
    float largestOutput = output[0];
    float sum_exp = 0.0f;

    for (size_t output_idx = 0; output_idx < NUM_OUTPUT_NEURONS; output_idx++)
    {
        if (output[output_idx] > largestOutput)
        {
            largestOutput = output[output_idx];
        }
    }

    memcpy(probabilities, output, NUM_OUTPUT_NEURONS * sizeof(*output));

    for (size_t idx = 0; idx < NUM_OUTPUT_NEURONS; idx++)
    {
        probabilities[idx] -= largestOutput;
        probabilities[idx] = expf(probabilities[idx]);
        sum_exp += probabilities[idx];
    }


    for (size_t idx = 0; idx < NUM_OUTPUT_NEURONS; idx++)
    {
        probabilities[idx] = probabilities[idx] / sum_exp;
    }
}



int main(void)
{
    FILE *train_images_file = fopen("mnist-dataset/train-images.idx3-ubyte", "rb");
    FILE *train_labels_file = fopen("mnist-dataset/train-labels.idx1-ubyte", "rb");

    if (train_images_file == NULL)
    {
        perror("Error opening train_images_file");
        return 1;
    }

    if (train_labels_file == NULL)
    {
        perror("Error opening train_labels_file");
        return 1;
    }

    // Buffer for images header
    unsigned char header[16];
    fread(header, 1, 16, train_images_file);

    Images images;

    uint32_t magic_number;

    // Stored in big-endian so have to read the bytes accordingly
   magic_number =
        (header[0] << 24) |
        (header[1] << 16) |
        (header[2] << 8)  |
        (header[3] << 0);

    // The MNIST dataset must have this magic number
    if (magic_number != 2051)
    {
        fprintf(stderr, "Invalid image file\n");
        return 1;
    }

    images.image_count =
        (header[4] << 24) |
        (header[5] << 16) |
        (header[6] << 8)  |
        (header[7] << 0);

    images.rows =
        (header[8] << 24) |
        (header[9] << 16) |
        (header[10] << 8) |
        (header[11] << 0);

    images.cols =
        (header[12] << 24) |
        (header[13] << 16) |
        (header[14] << 8)  |
        (header[15] << 0);

    // Buffer for labels header
    unsigned char label_header[8];
    fread(label_header, 1, 8, train_labels_file);

    // Initializing labels header variables
    uint32_t label_magic_number;
    uint32_t label_count;

    label_magic_number =
        (label_header[0] << 24) |
        (label_header[1] << 16) |
        (label_header[2] << 8)  |
        (label_header[3]);

    if (label_magic_number != 2049)
    {
        fprintf(stderr, "Invalid label file\n");
        return 1;
    }

    label_count =
        (label_header[4] << 24) |
        (label_header[5] << 16) |
        (label_header[6] << 8)  |
        (label_header[7]);

    if (images.image_count != label_count)
    {
        fprintf(stderr, "Image count and label count do not match\n");
        return 1;
    }


    // Total pixel bytes: 60,000 images * 28 * 28 pixels = 47,040,000 bytes
    size_t read_size = (size_t)images.image_count * images.rows * images.cols;


    // Allocating and storing all images in memory
    images.data = calloc(read_size, 1);
    fread(images.data, 1, read_size, train_images_file);

    // Allocating and storing all labels in memory
    uint8_t *labels = calloc(label_count, sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), label_count, train_labels_file);

    fclose(train_images_file);
    fclose(train_labels_file);

    // Load test images
    FILE *test_images_file = fopen("mnist-dataset/t10k-images.idx3-ubyte", "rb");
    FILE *test_labels_file = fopen("mnist-dataset/t10k-labels.idx1-ubyte", "rb");

    if (test_images_file == NULL)
    {
        perror("Error opening test_images_file");
        return 1;
    }

    if (test_labels_file == NULL)
    {
        perror("Error opening test_labels_file");
        return 1;
    }

    // Buffer for test images header
    unsigned char test_header[16];
    fread(test_header, 1, 16, test_images_file);

    Images test_images;

    uint32_t test_magic_number =
        (test_header[0] << 24) |
        (test_header[1] << 16) |
        (test_header[2] << 8)  |
        (test_header[3] << 0);

    if (test_magic_number != 2051)
    {
        fprintf(stderr, "Invalid test image file\n");
        return 1;
    }

    test_images.image_count =
        (test_header[4] << 24) |
        (test_header[5] << 16) |
        (test_header[6] << 8)  |
        (test_header[7] << 0);

    test_images.rows =
        (test_header[8] << 24) |
        (test_header[9] << 16) |
        (test_header[10] << 8) |
        (test_header[11] << 0);

    test_images.cols =
        (test_header[12] << 24) |
        (test_header[13] << 16) |
        (test_header[14] << 8)  |
        (test_header[15] << 0);

    // Buffer for test labels header
    unsigned char test_label_header[8];
    fread(test_label_header, 1, 8, test_labels_file);

    uint32_t test_label_magic_number =
        (test_label_header[0] << 24) |
        (test_label_header[1] << 16) |
        (test_label_header[2] << 8)  |
        (test_label_header[3]);

    if (test_label_magic_number != 2049)
    {
        fprintf(stderr, "Invalid test label file\n");
        return 1;
    }

    uint32_t test_label_count =
        (test_label_header[4] << 24) |
        (test_label_header[5] << 16) |
        (test_label_header[6] << 8)  |
        (test_label_header[7]);

    if (test_images.image_count != test_label_count)
    {
        fprintf(stderr, "Test image count and label count do not match\n");
        return 1;
    }

    // 10,000 images * 28 * 28 pixels = 7,840,000 bytes
    size_t test_read_size = (size_t)test_images.image_count * test_images.rows * test_images.cols;

    test_images.data = calloc(test_read_size, 1);
    fread(test_images.data, 1, test_read_size, test_images_file);

    uint8_t *test_labels = calloc(test_label_count, sizeof(uint8_t));
    fread(test_labels, sizeof(uint8_t), test_label_count, test_labels_file);

    fclose(test_images_file);
    fclose(test_labels_file);

    // Allocate one weight per input connection:
    // each pixel has a separate weight for each hidden neuron,
    // and each hidden neuron has a separate weight for each output neuron

    float *weights1 = malloc(NUM_HIDDEN_NEURONS * images.rows * images.cols * sizeof(*weights1));
    float *weights2 = malloc(NUM_OUTPUT_NEURONS * NUM_HIDDEN_NEURONS * sizeof(*weights2));

    // Setting random weights1
    for (size_t neuron = 0; neuron < NUM_HIDDEN_NEURONS; neuron++)
    {
        for (size_t pixel = 0; pixel < images.rows * images.cols; pixel++)
        {
            weights1[neuron * (images.rows * images.cols) + pixel] = ((float)rand() / RAND_MAX) * 0.1 - 0.05;
        }
    }

    // Setting random weights2
    for (size_t w2 = 0; w2 < NUM_OUTPUT_NEURONS * NUM_HIDDEN_NEURONS; w2++)
    {
        weights2[w2] = ((float)rand() / RAND_MAX) * 0.1 - 0.05;
    }


    // Hidden-layer activations (outputs after ReLU)
    float activation1[NUM_HIDDEN_NEURONS] = {0};

    // Final output-layer scores and softmax probabilities
    float output[NUM_OUTPUT_NEURONS] = {0};
    float probabilities[NUM_OUTPUT_NEURONS] = {0};

    // One trainable bias per hidden/output neuron
    float bias1[NUM_HIDDEN_NEURONS] = {0};
    float bias2[NUM_OUTPUT_NEURONS] = {0};

    // Gradients for the output layer
    float output_gradients[NUM_OUTPUT_NEURONS] = {0};
    float weights2_gradients[NUM_OUTPUT_NEURONS * NUM_HIDDEN_NEURONS] = {0};
    float bias2_gradients[NUM_OUTPUT_NEURONS] = {0};

    // Gradients flowing back into the hidden layer
    float activation1_gradients[NUM_HIDDEN_NEURONS] = {0};

    size_t pixels_per_image = images.rows * images.cols;

    float *weights1_gradients = malloc(NUM_HIDDEN_NEURONS * pixels_per_image * sizeof(*weights1_gradients));
    float bias1_gradients[NUM_HIDDEN_NEURONS] = {0};

    float total_loss = 0.0f;
    int correct = 0;

    // Runs forward pass, calculates loss and gradient
    for (size_t image_idx = 0; image_idx < images.image_count; image_idx++)
    {
        forward_pass(&images, image_idx, weights1, weights2, activation1, output, bias1, bias2, probabilities);
        float loss = -logf(probabilities[labels[image_idx]]);

        total_loss += loss;

        int predicted = 0;

        for (int i = 1; i < NUM_OUTPUT_NEURONS; i++)
        {
            if (probabilities[i] > probabilities[predicted])
            {
                predicted = i;
            }
        }

        if (predicted == labels[image_idx])
        {
            correct++;
        }

        // Every 5000 images, print and reset
        if ((image_idx + 1) % 5000 == 0)
        {
            printf("Images %zu: avg loss %.4f, accuracy %.2f%%\n", image_idx + 1, total_loss / 5000, 100.0f * correct / 5000);
            total_loss = 0.0f;
            correct = 0;
        }

        memcpy(output_gradients, probabilities, NUM_OUTPUT_NEURONS * sizeof(*probabilities));
        output_gradients[labels[image_idx]] -= 1;

        for (int output_idx = 0; output_idx < NUM_OUTPUT_NEURONS; output_idx++)
        {
            bias2_gradients[output_idx] = output_gradients[output_idx];


            for (int hidden_idx = 0; hidden_idx < NUM_HIDDEN_NEURONS; hidden_idx++)
            {
                size_t weight_idx = output_idx * NUM_HIDDEN_NEURONS + hidden_idx;

                weights2_gradients[weight_idx] = output_gradients[output_idx] * activation1[hidden_idx];
            }
        }

        for (int hidden_idx = 0; hidden_idx < NUM_HIDDEN_NEURONS; hidden_idx++)
        {
            activation1_gradients[hidden_idx] = 0;

            for (int output_idx = 0; output_idx < NUM_OUTPUT_NEURONS; output_idx++)
            {
                size_t weight_idx = output_idx * NUM_HIDDEN_NEURONS + hidden_idx;
                activation1_gradients[hidden_idx] += output_gradients[output_idx] * weights2[weight_idx];
            }
        }

        for (int hidden_idx = 0; hidden_idx < NUM_HIDDEN_NEURONS; hidden_idx++)
        {
            if (activation1[hidden_idx] <= 0)
            {
                activation1_gradients[hidden_idx] = 0;
            }
        }

        size_t image_start = image_idx * pixels_per_image;

        for (int hidden_idx = 0; hidden_idx < NUM_HIDDEN_NEURONS; hidden_idx++)
        {
            bias1_gradients[hidden_idx] = activation1_gradients[hidden_idx];

            for (size_t pixel_idx = 0; pixel_idx < pixels_per_image; pixel_idx++)
            {
                float normalized = images.data[image_start + pixel_idx] / 255.0f;
                weights1_gradients[hidden_idx * pixels_per_image + pixel_idx] = activation1_gradients[hidden_idx] * normalized;
            }
        }

        for (size_t i = 0; i < NUM_OUTPUT_NEURONS * NUM_HIDDEN_NEURONS; i++)
        {
            weights2[i] -= LEARNING_RATE * weights2_gradients[i];
        }

        for (size_t i = 0; i < NUM_OUTPUT_NEURONS; i++)
        {
            bias2[i] -= LEARNING_RATE * bias2_gradients[i];
        }

        for (size_t i = 0; i < NUM_HIDDEN_NEURONS * pixels_per_image; i++)
        {
            weights1[i] -= LEARNING_RATE * weights1_gradients[i];
        }

        for (size_t i = 0; i < NUM_HIDDEN_NEURONS; i++)
        {
            bias1[i] -= LEARNING_RATE * bias1_gradients[i];
        }
    }


    // Evaluate on unseen test images using the trained weights
    int test_correct = 0;
    float test_total_loss = 0.0f;

    for (size_t image_idx = 0; image_idx < test_images.image_count; image_idx++)
    {
        forward_pass(&test_images, image_idx, weights1, weights2, activation1, output, bias1, bias2, probabilities);

        test_total_loss += -logf(probabilities[test_labels[image_idx]]);

        int predicted = 0;
        for (int i = 1; i < NUM_OUTPUT_NEURONS; i++)
        {
            if (probabilities[i] > probabilities[predicted])
            {
                predicted = i;
            }
        }

        if (predicted == test_labels[image_idx])
        {
            test_correct++;
        }
    }

    printf("\nTest set: avg loss %.4f, accuracy %.2f%% (%d / %u)\n",
           test_total_loss / test_images.image_count,
           100.0f * test_correct / test_images.image_count,
           test_correct, test_images.image_count);
}
