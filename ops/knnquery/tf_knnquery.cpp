
#include <cuda_runtime.h>

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("KnnQuery")
    .Input("queries: float32")
    .Input("supports: float32")
    .Input("q_offset: int32")
    .Input("s_offset: int32")
    .Input("k: int32")
    .Output("out: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Create input shape container
        ::tensorflow::shape_inference::ShapeHandle input_shape;

        // Check inputs rank
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));  // query shape

        // Create the output shape
        ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(input_shape, 0), -1});  // [BxNq, ?]
        c->set_output(0, output_shape);

        return Status::OK();
    });

void knnquery_cuda_launcher(int m, int nsample, const float* xyz, const float* new_xyz, const int* offset, const int* new_offset, int* idx, float* dist2);  // include using .o

class KnnQueryGpuOp : public OpKernel {
   public:
    // constructor
    explicit KnnQueryGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // tf input
        const Tensor& queries_tensor = context->input(0);
        const Tensor& supports_tensor = context->input(1);
        const Tensor& q_offset_tensor = context->input(2);
        const Tensor& s_offset_tensor = context->input(3);
        const Tensor& k_tensor = context->input(4);

        // size info & check
        const TensorShape& queries_shape = queries_tensor.shape();
        const TensorShape& supports_shape = supports_tensor.shape();
        const TensorShape& q_offset_shape = q_offset_tensor.shape();
        const TensorShape& s_offset_shape = s_offset_tensor.shape();

        OP_REQUIRES(context, queries_shape.dims() == 2 && queries_shape.dim_size(1) == 3, errors::InvalidArgument("KnnQuery expects 'queries' to be of shape [BxN, 3]"));
        OP_REQUIRES(context, supports_shape.dims() == 2 && supports_shape.dim_size(1) == 3, errors::InvalidArgument("KnnQuery expects 'supports' to be of shape [BxN, 3]"));
        OP_REQUIRES(context, q_offset_shape.dims() == 1, errors::InvalidArgument("KnnQuery expects 'q_offset' to be of shape [B]"));
        OP_REQUIRES(context, s_offset_shape.dims() == 1, errors::InvalidArgument("KnnQuery expects 's_offset' to be of shape [B]"));
        OP_REQUIRES(context, q_offset_shape.dim_size(0) == s_offset_shape.dim_size(0), errors::InvalidArgument("KnnQuery expects same #examples in queries and supports"));

        const int B = q_offset_shape.dim_size(0);
        const int q_BN = queries_shape.dim_size(0);  // parallel on per-point querying
        // const int s_BN = supports_shape.dim_size(0);
        // ::std::cout << "B " << B << " q_BN " << q_BN << ::std::endl;

        // tf -> cpp
        // - tensors already on gpu
        const float* queries = queries_tensor.flat<float>().data();
        const float* supports = supports_tensor.flat<float>().data();
        const int* q_offset = q_offset_tensor.flat<int>().data();
        const int* s_offset = s_offset_tensor.flat<int>().data();
        // ::std::cout << "ptr - q/s " << queries << " " << supports << ::std::endl;

        // tf -> cpp scalar
        // - k on cpu (as noted by HostMemory)
        // ::std::cout << "k - #elem " << k_tensor.NumElements() << " dims " << k_tensor.dims() << " scalar " << k_tensor.scalar<int>() << ::std::endl;
        // - k_tensor.scalar<int>() = Eigen::TensorMap<Eigen::TensorFixedSize<const int, Eigen::Sizes<>, 1, long int>, 16, Eigen::MakePointer>
        const int k = int(k_tensor.scalar<int>()());
        // ::std::cout << "k " << k << ::std::endl;


        Tensor* neighbors_idx_tensor;  // output - neighbor_idx
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{q_BN, k}, &neighbors_idx_tensor));
        int* neighbors_idx = neighbors_idx_tensor->flat<int>().data();

        Tensor neighbor_dist_tensor;  // temp - dist in l2-square
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{q_BN, k}, &neighbor_dist_tensor));
        float* dist2 = &(neighbor_dist_tensor.flat<float>()(0));  // cannot use .data(), why?

        knnquery_cuda_launcher(q_BN, k, supports, queries, s_offset, q_offset, neighbors_idx, dist2);
    }
};

REGISTER_KERNEL_BUILDER(Name("KnnQuery")
                            .Device(DEVICE_GPU)
                            /* store inputs in cpu mem: k */
                            .HostMemory("k"),
                        KnnQueryGpuOp);
