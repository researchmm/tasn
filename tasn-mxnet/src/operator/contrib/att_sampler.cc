/*!
 * \file att_sampler.cc
 * \author Heliang Zheng
 * \adapted from https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu
*/


#include "./att_sampler-inl.h"
namespace mshadow {
 template<typename DType>
 inline void AttSamplerForward(const Tensor<cpu, 4, DType> &output,
  const Tensor<cpu, 4, DType> &input,
  const Tensor<cpu, 4, DType> &grid_src) {
  LOG(INFO) << "cpu version not implemented";
 }

 template<typename DType>
 inline void AttSamplerBackward(const Tensor<cpu, 4, DType> &input_grad,
  const Tensor<cpu, 4, DType> &output_grad,
  const Tensor<cpu, 4, DType> &input_data,
  const Tensor<cpu, 4, DType> &grid) {

  LOG(INFO) << "cpu version not implemented";
 }
}  // namespace mshadow

namespace mxnet {
 namespace op {
  template<>
  Operator* CreateOp<cpu>(AttSamplerParam param, int dtype) {
   Operator *op = NULL;
   MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AttSamplerOp<cpu, DType>(param);
   })
    return op;
  }

  Operator *AttSamplerProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
   std::vector<int> *in_type) const {
   DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
  }

  DMLC_REGISTER_PARAMETER(AttSamplerParam);

  MXNET_REGISTER_OP_PROPERTY(_contrib_AttSampler, AttSamplerProp)
   .add_argument("data", "NDArray-or-Symbol", "Input data to the AttSamplerOp.")
   .add_argument("attx", "NDArray-or-Symbol", "Input attx to the AttSamplerOp.")
   .add_argument("atty", "NDArray-or-Symbol", "Input atty to the AttSamplerOp.")
   .add_arguments(AttSamplerParam::__FIELDS__());
 }  // namespace op
}  // namespace mxnet
