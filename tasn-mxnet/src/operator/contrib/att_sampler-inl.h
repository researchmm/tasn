/*!
 * \file att_sampler-inl.h
 * \author Heliang Zheng
 * \adapted from https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler-inl.h
*/


#ifndef MXNET_OPERATOR_ATT_SAMPLER_INL_H_
#define MXNET_OPERATOR_ATT_SAMPLER_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "../../common/utils.h"
namespace mxnet {
 namespace op {

  namespace ag {
   enum AttSamplerOpInputs { kData, kAttx, kAtty };
   enum AttSamplerOpOutputs { kOut, kTmp };
  }  // namespace crop_enum

  struct AttSamplerParam : public dmlc::Parameter<AttSamplerParam> {
   float scale;
   float dense;
   int iters;
   DMLC_DECLARE_PARAMETER(AttSamplerParam) {
    DMLC_DECLARE_FIELD(scale).set_default(1.0)
     .describe("The ratio of input size and output size.");
    DMLC_DECLARE_FIELD(dense).set_default(4)
     .describe("The max amplification.");
    DMLC_DECLARE_FIELD(iters).set_default(5)
     .describe("Iterations for ratio normalization.");
   }
  };  // struct attParam

  template<typename xpu, typename DType>
  class AttSamplerOp : public Operator {
  public:
   explicit AttSamplerOp(AttSamplerParam param) {
    this->param_ = param;
   }

   virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Stream<cpu> *cpu_s = ctx.get_stream<cpu>();

    Tensor<xpu, 3, DType> attx= in_data[ag::kAttx].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> atty= in_data[ag::kAtty].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> grid = out_data[ag::kTmp].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[ag::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[ag::kOut].get<xpu, 4, DType>(s);
    int n = data.size(0);
    int in_size = data.size(2);
    int att_size = attx.size(1);
    int out_size = in_size * param_.scale;

    Tensor<cpu, 4, DType> cpu_grid = NewTensor<cpu>(grid.shape_, DType(1.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_x = NewTensor<cpu>(Shape3(n,att_size,1), DType(0.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_xi = NewTensor<cpu>(Shape3(n,att_size,1), DType(1.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_y = NewTensor<cpu>(Shape3(n,att_size,1), DType(0.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_yi = NewTensor<cpu>(Shape3(n,att_size,1), DType(1.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_sum_x = NewTensor<cpu>(Shape3(n, 1, 1), DType(1.0), false, cpu_s);
    Tensor<cpu, 3, DType> cpu_map_sum_y = NewTensor<cpu>(Shape3(n, 1, 1), DType(1.0), false, cpu_s);
    Tensor<cpu, 3, DType> index_x = NewTensor<cpu>(Shape3(n, out_size, 1), DType(0.0), false, cpu_s);
    Tensor<cpu, 3, DType> index_y = NewTensor<cpu>(Shape3(n, out_size, 1), DType(0.0), false, cpu_s);
    Tensor<cpu, 2, DType> cpu_one_vector_att = NewTensor<cpu>(Shape2(1, att_size), DType(1.0), false, cpu_s);
    Tensor<cpu, 2, DType> cpu_one_vector = NewTensor<cpu>(Shape2(1, out_size), DType(1.0), false, cpu_s);
    Tensor<xpu, 2, DType> one_vector = NewTensor<xpu>(Shape2(1, out_size), DType(1.0), false, s);

    Copy(cpu_map_x,attx, s);
    Copy(cpu_map_y,atty, s);

    DType *mapx_ = cpu_map_x.dptr_;
    DType *mapxi_ = cpu_map_xi.dptr_;
    DType *mapy_ = cpu_map_y.dptr_;
    DType *mapyi_ = cpu_map_yi.dptr_;
    DType *index_x_ = index_x.dptr_;
    DType *index_y_ = index_y.dptr_;
    DType threshold = param_.scale * param_.dense * in_size / att_size;
    DType threshold_in_use = param_.scale * param_.dense * in_size / att_size;
    DType *mapsumx_ = cpu_map_sum_x.dptr_;
    DType *mapsumy_ = cpu_map_sum_y.dptr_;
    DType map_maxx;
    DType map_maxy;
    DType sumx;
    DType sumy;
    DType deltax;
    DType deltay;
    int step;
    for (index_t b = 0; b < n; ++b) {
     step = b * att_size;
     DType* mapx = mapx_ + b * att_size;
     DType* mapy = mapy_ + b * att_size;
     for (int i = 0; i < att_size; i++)
     {
      mapx_[step + i] = mapx_[step + i] * out_size;
      mapy_[step + i] = mapy_[step + i] * out_size;
     }
     for (int j = 0; j < param_.iters; j++)
     {
      map_maxx = DType(0);
      map_maxy = DType(0);
      for (int i = 0; i < att_size; i++)
      {
       map_maxx = map_maxx > mapx_[step + i] ? map_maxx : mapx_[step + i];
       map_maxy = map_maxy > mapy_[step + i] ? map_maxy : mapy_[step + i];
      }
      map_maxx = map_maxx > map_maxy ? map_maxy : map_maxx;
      threshold_in_use = map_maxx;
      if (j==0)
       threshold_in_use = threshold > map_maxx ? map_maxx : threshold;
      for (int i = 0; i < att_size; i++)
      {
       mapx_[step + i] = mapx_[step + i] > threshold_in_use ? threshold_in_use : mapx_[step + i];
       mapy_[step + i] = mapy_[step + i] > threshold_in_use ? threshold_in_use : mapy_[step + i];
      }

      linalg_gemm(cpu_map_x[b], cpu_one_vector_att, cpu_map_sum_x[b], DType(1), DType(0), true, true, cpu_s);
      linalg_gemm(cpu_map_y[b], cpu_one_vector_att, cpu_map_sum_y[b], DType(1), DType(0), true, true, cpu_s);

      sumx = *(mapsumx_ + b);
      sumy = *(mapsumy_ + b);
      deltax = (out_size - sumx) / att_size;
      deltay = (out_size - sumy) / att_size;
      for (int i = 0; i < att_size; i++)
      {
       mapx[i] += deltax;
       mapy[i] += deltay;
      }
     }


    }

    for (index_t b = 0; b < n; ++b) {
     DType* mapx = mapx_ + b * att_size;
     DType* mapxi = mapxi_ + b * att_size;
     DType* mapy = mapy_ + b * att_size;
     DType* mapyi = mapyi_ + b * att_size;
     DType* indexy = index_y_ + b * out_size;
     DType* indexx = index_x_ + b * out_size;
     for (int i = 0; i < att_size - 1; i++)
     {
      mapxi[i + 1] = mapxi[i] + mapx[i + 1];
      mapyi[i + 1] = mapyi[i] + mapy[i + 1];
     }
     DType stepx = mapxi[att_size - 1] / out_size;
     DType stepy = mapyi[att_size - 1] / out_size;
     int i = 0;
     int j = 0;
     DType myscale = 2.0 / (att_size - 1);
     while (i<out_size)
     {
      if (mapxi[j] >= i*stepx)
      {
       indexy[i] = (j + (i*stepx - mapxi[j]) / (mapxi[j] - mapxi[j - 1])) * myscale - 1.0;
       i++;
      }
      else {
       j++;
      }
     }
     i = 0;
     j = 0;
     while (i<out_size)
     {
      if (mapyi[j] >= i*stepy)
      {
       indexx[i] = (j + (i*stepy - mapyi[j]) / (mapyi[j] - mapyi[j - 1])) * myscale - 1.0;
       i++;
      }
      else {
       j++;
      }
     }

    }

    for (index_t i = 0; i < data.size(0); ++i) {
     linalg_gemm(cpu_one_vector, index_x[i], cpu_grid[i][0], DType(1), DType(0), true, true, cpu_s);
     linalg_gemm(index_y[i], cpu_one_vector, cpu_grid[i][1], DType(1), DType(0), false, false, cpu_s);
    }
    Copy(grid, cpu_grid, s);
    AttSamplerForward(out, data, grid);
    FreeSpace(&cpu_grid);
    FreeSpace(&cpu_map_x);
    FreeSpace(&cpu_map_xi);
    FreeSpace(&cpu_map_y);
    FreeSpace(&cpu_map_yi);
    FreeSpace(&cpu_map_sum_x);
    FreeSpace(&cpu_map_sum_y);
    FreeSpace(&index_x);
    FreeSpace(&index_y);
    FreeSpace(&cpu_one_vector);
    FreeSpace(&cpu_one_vector_att);
    FreeSpace(&one_vector);


   }


   virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob> &out_grad,
    const std::vector<TBlob> &in_data,
    const std::vector<TBlob> &out_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &in_grad,
    const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> gdata = in_grad[ag::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[ag::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> gattx = in_grad[ag::kAttx].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> gatty = in_grad[ag::kAtty].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> grid = out_data[ag::kTmp].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[ag::kOut].get<xpu, 4, DType>(s);
    gattx = scalar<DType>(0.0f);
    gatty = scalar<DType>(0.0f);
    gdata = scalar<DType>(0.0f);
    AttSamplerBackward(gdata, grad, data, grid);

   }

  private:
   AttSamplerParam param_;
  };  // class AttSamplerOp
  template<typename xpu>
  Operator* CreateOp(AttSamplerParam param, int dtype);

#if DMLC_USE_CXX11
  class AttSamplerProp : public OperatorProperty {
  public:
   void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
   }

   std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
   }
   int NumVisibleOutputs() const override {
    return 1;
   }

   int NumOutputs() const override {
    return 2;
   }
   std::vector<std::string> ListArguments() const override {
    return{ "data", "attx", "atty" };
   }

   std::vector<std::string> ListOutputs() const override {
    return{ "output", "tmp"};
   }

   bool InferShape(std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    const TShape &dshape = (*in_shape)[ag::kData];
    int output_size = int(dshape[2] * param_.scale);//static_cast<int>(ceil(static_cast<float>(dshape[2] * param_.scale)));
    out_shape->clear();
    out_shape->push_back(Shape4(dshape[0], dshape[1], output_size, output_size));
    //aux_shape->clear();
    out_shape->push_back(Shape4(dshape[0], 2, output_size, output_size));

    return true;
   }

   OperatorProperty* Copy() const override {
    auto ptr = new AttSamplerProp();
    ptr->param_ = param_;
    return ptr;
   }

   std::string TypeString() const override {
    return "_contrib_AttSampler";
   }

   std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[ag::kOut],
     out_data[ag::kTmp],
     in_data[ag::kData] };
   }

   Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
   }

   Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;

   bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    int dtype = -1;
    for (size_t i = 0; i < in_type->size(); ++i) {
     if (dtype == -1) {
      dtype = in_type->at(i);
     }
     else {
      CHECK(in_type->at(i) == dtype ||
       in_type->at(i) == -1) <<
       "Non-uniform data type in AttSampler";
     }
    }
    if (dtype == -1) {
     LOG(FATAL) << "Not enough information to infer type in AttSampler.";
     return false;
    }
    size_t nin = this->ListArguments().size();
    in_type->clear();
    for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);
    size_t naux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);
    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
   }

  private:
   AttSamplerParam param_;
  };
#endif  // DMLC_USE_CXX11
 }  // namespace op
}  // namespace mxnet
#endif 
