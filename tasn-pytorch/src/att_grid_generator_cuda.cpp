#include <torch/extension.h>

#include <cmath>
#include <vector>

void attgridgen_gpu(at::Tensor attx, at::Tensor atty,
                    at::Tensor map_xi, at::Tensor map_yi,
                    at::Tensor index_x, at::Tensor index_y,
                    const int batch_size, const int att_size, const int out_size,
                    const float threshold, const int iters);

void shape_check(at::Tensor attx, at::Tensor atty, int batch_size, int att_size)
{
    AT_CHECK(attx.ndimension() == 3,
             "4D attx map tensor (batchSize, att_size, 1),",
             "but got: %s",
             attx.ndimension());
    AT_CHECK(attx.is_contiguous(), "attention_x tentor has to be contiguous");

    AT_CHECK(att_size > 0,
             "attention size should be greater than zero, but got att_size: %d", att_size);
    AT_CHECK(batch_size > 0,
             "batch size should be greater than zero, but got att_size: %d", batch_size);

    AT_CHECK(attx.size(1) == batch_size,
             "att_size should be consistent with attention_x,",
             "but got batch_size: %d, attx.size(1): %d", att_size, attx.size(1));
    AT_CHECK(attx.size(2) == att_size,
             "att_size should be consistent with attention_x,",
             "but got att_size: %d, att.size(2): %d", att_size, attx.size(2));
    AT_CHECK(atty.ndimension() == 3,
             "4D atty map tensor (batchSize, att_size, 1),",
             "but got: %s",
             atty.ndimension());
    AT_CHECK(atty.is_contiguous(), "attention_y tentor has to be contiguous");

    AT_CHECK(atty.size(1) == batch_size,
             "att_size should be consistent with attention_y,",
             "but got batch_size: %d, atty.size(1): %d", att_size, atty.size(1));
    AT_CHECK(atty.size(2) == att_size,
             "att_size should be consistent with attention_y,",
             "but got att_size: %d, atty.size(2): %d", att_size, atty.size(2));
}

int forward(at::Tensor attx, at::Tensor atty,
            at::Tensor map_xi, at::Tensor map_yi,
            at::Tensor index_x, at::Tensor index_y,
            //at::Tensor grid,
            //int batch_size, int att_size,
            int input_size, int out_size, int dense, int iters, float scale)
{
    attx = attx.contiguous();
    atty = atty.contiguous();

    //shape_check(attx, atty, batch_size, att_size);
    long batch_size = attx.size(0);
    long att_size = attx.size(1);
    float threshold = scale * dense * input_size / att_size;

    //at::Tensor map_xi = at::ones({batch_size, att_size, 1}, attx.options());
    //at::Tensor map_yi = at::ones({batch_size, att_size, 1}, atty.options());
    //at::Tensor map_sum_x = at::ones({batch_size, 1, 1}, attx.options());
    //at::Tensor map_sum_y = at::ones({batch_size, 1, 1}, atty.options());

    //at::Tensor index_x = at::ones({batch_size, out_size, 1}, attx.options());
    //at::Tensor index_y = at::ones({batch_size, out_size, 1}, atty.options());

    attgridgen_gpu(attx, atty, map_xi, map_yi, index_x, index_y,
                   batch_size, att_size, out_size, threshold, iters);

    index_x = index_x.view({batch_size, out_size, 1});
    index_y = index_y.view({batch_size, out_size, 1});
    //attx = attx.view({batch_size, input_size, 1});
    //atty = atty.view({batch_size, input_size, 1});

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward,
          "attgridgenerator forward (CUDA)");
}
