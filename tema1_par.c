#define _XOPEN_SOURCE 700

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

typedef struct {
    ppm_image *image, *scaled_image;
    int p, q, step_x, step_y, thread_id, num_threads;
    unsigned char **grid;
    ppm_image **contour_map;
    pthread_barrier_t barrier;
} thread_data;

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}


void *thread_function(void *arg) {

    // rescale image
    uint8_t sample[3];

    if (arg == NULL) {
        fprintf(stderr, "Thread argument is NULL\n");
        exit(1);
    }

    thread_data *data = (thread_data *)arg;

    if (data->image->x <= RESCALE_X && data->image->y <= RESCALE_Y) {
        data->scaled_image = data->image;
    } else {
        data->scaled_image->x = RESCALE_X;
        data->scaled_image->y = RESCALE_Y;
        int start = data->thread_id * data->scaled_image->x / data->num_threads;
        int end = fmin((data->thread_id + 1) * data->scaled_image->x / data->num_threads, data->scaled_image->x);

        for (int i = start; i < end; i++) {
            for (int j = 0; j < data->scaled_image->y; j++) {
                float u = (float)i / (float)(data->scaled_image->x - 1);
                float v = (float)j / (float)(data->scaled_image->y - 1);
                sample_bicubic(data->image, u, v, sample);

                data->scaled_image->data[i * data->scaled_image->y + j].red = sample[0];
                data->scaled_image->data[i * data->scaled_image->y + j].green = sample[1];
                data->scaled_image->data[i * data->scaled_image->y + j].blue = sample[2];
            }
        }
        
        // pthread_barrier_wait(&(data->barrier));
        // printf("adresa2: %p\n", &data->barrier);

        // free(data->image->data);
        // free(data->image);
    }
    // pthread_barrier_wait(&(data->barrier));
    // printf("adresa2: %p\n", &data->barrier);

    // sample grid

    int start = data->thread_id * (double) data->p / data->num_threads;
    int end = fmin((data->thread_id + 1) * (double) data->p / data->num_threads, data->p);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < data->q; j++) {
            ppm_pixel curr_pixel = data->scaled_image->data[i * data->step_x * data->scaled_image->y + j * data->step_y];
            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA) {
                data->grid[i][j] = 0;
            } else {
                data->grid[i][j] = 1;
            }
        }
                
    }
    data->grid[data->p][data->q] = 0;
    // pthread_barrier_wait(&data->barrier);

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them

    for (int i = 0; i < data->p; i++) {
        ppm_pixel curr_pixel = data->scaled_image->data[i * data->step_x * data->scaled_image->y + data->scaled_image->x - 1];
        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            data->grid[i][data->q] = 0;
        } else {
            data->grid[i][data->q] = 1;
        }
    }
    for (int j = 0; j < data->q; j++) {
        ppm_pixel curr_pixel = data->scaled_image->data[(data->scaled_image->x - 1) * data->scaled_image->y + j * data->step_y];
        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            data->grid[data->p][j] = 0;
        } else {
            data->grid[data->p][j] = 1;
        }
    }

    // march

    for (int i = start; i < end; i++) {
        for (int j = 0; j < data->q; j++) {
            unsigned char k = 8 * data->grid[i][j] + 4 * data->grid[i][j + 1] + 2 * data->grid[i + 1][j + 1] + 1 * data->grid[i + 1][j];
            update_image(data->scaled_image, data->contour_map[k], i * data->step_x, j * data->step_y);
        }
    }
}



int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;
    int p = image->x / step_x;
    int q = image->y / step_y;

    int i, r;
    void *status;
    int num_threads = atoi(argv[3]);
    pthread_t threads[num_threads];

    // allocate memory
    thread_data *data = (thread_data *)malloc(num_threads * sizeof(thread_data));
    if (data == NULL) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    ppm_image *scaled_image = malloc(sizeof(ppm_image));
    if (scaled_image == NULL) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    scaled_image->x = RESCALE_X;
    scaled_image->y = RESCALE_Y;
    scaled_image->data = (ppm_pixel*)malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (grid == NULL) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    for (i = 0; i <= p; i++) {
        grid[i] = malloc((q + 1) * sizeof(unsigned char));
        if (grid[i] == NULL) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    ppm_image **contour_map = init_contour_map();
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);
    printf("adresa1: %p\n", &barrier);


    for (i = 0; i < num_threads; i++) {
        data[i].image = image;
        data[i].scaled_image = scaled_image;
        data[i].p = p;
        data[i].q = q;
        data[i].grid = grid;
        data[i].contour_map = contour_map;
        data[i].step_x = step_x;
        data[i].step_y = step_y;
        data[i].thread_id = i;
        data[i].num_threads = num_threads;
        data[i].barrier = barrier;

        r = pthread_create(&threads[i], NULL, thread_function, (void *)&data[i]);
        if (r) {
            fprintf(stderr, "Error creating thread\n");
            exit(1);
        }
    }

    for (i = 0; i < num_threads; i++) {
        r = pthread_join(threads[i], &status);
        if (r) {
            fprintf(stderr, "Error joining thread\n");
            exit(1);
        }
    }

    // 4. Write output
    //write_ppm(scaled_image, argv[2]);
    write_ppm(data[0].scaled_image, argv[2]);

    // free(data);

    // free_resources(scaled_image, contour_map, grid, step_x);

    return 0;
}
