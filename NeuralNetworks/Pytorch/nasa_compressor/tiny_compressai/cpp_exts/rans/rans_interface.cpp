/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <torch/script.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}

inline int32_t to_int32(int64_t i){
    return (int32_t)i;
}

inline int64_t to_int64(int32_t i){
    return (int64_t)i;
}
} // namespace

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value < cdfs_sizes[cdf_idx] - 1);

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

std::string BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}


RansEncoder::RansEncoder(int64_t init) : _state(init){}

torch::Tensor
RansEncoder::encode_with_indexes(const std::vector<int64_t> &symbols_i,
                                 const std::vector<int64_t> &indexes_i,
                                 const std::vector<std::vector<int64_t>> &cdfs_i,
                                 const std::vector<int64_t> &cdfs_sizes_i,
                                 const std::vector<int64_t> &offsets_i) {

	//std::cout << "RansEncoder::encode_with_indexes START" << std::endl;

	//transform inputs to original types
	std::vector<int32_t> symbols(symbols_i.size());
	std::vector<int32_t> indexes(indexes_i.size());
	std::vector<std::vector<int32_t>> cdfs(cdfs_i.size());
	std::vector<int32_t> cdfs_sizes(cdfs_sizes_i.size());
	std::vector<int32_t> offsets(offsets_i.size());
	std::transform(symbols_i.begin(), symbols_i.end(), symbols.begin(), to_int32);
	std::transform(indexes_i.begin(), indexes_i.end(), indexes.begin(), to_int32);
	for(int i=0; i<cdfs_i.size(); i++ ){
		std::vector<int64_t> v = cdfs_i[i];
		cdfs[i] = std::vector<int32_t>(v.size());
		std::transform(v.begin(), v.end(), cdfs[i].begin(), to_int32);
	}
	std::transform(cdfs_sizes_i.begin(), cdfs_sizes_i.end(), cdfs_sizes.begin(), to_int32);
	std::transform(offsets_i.begin(), offsets_i.end(), offsets.begin(), to_int32);

	//std::cout << "symbols: " << symbols_i << "| " << symbols<< std::endl;
	//std::cout << "indexes: " << indexes_i << "| " << indexes<< std::endl;
	//std::cout << "cdfs: ";
	//for(int i=0; i<cdfs_i.size(); i++ ){
	//	std::cout << cdfs_i[i] << "| " << cdfs[i] << std::endl;
	//}
	//std::cout << "cdfs_sizes: " << cdfs_sizes_i << "| " << cdfs_sizes<< std::endl;
	//std::cout << "offsets: " << offsets_i << "| " << offsets<< std::endl;

	//std::cout << "symbols: " << symbols.size()<< std::endl;
	//std::cout << "indexes: " << indexes.size()<< std::endl;
	//std::cout << "cdfs: ";
	//for(int i=0; i<cdfs_i.size(); i++ ){
	//	std::cout << cdfs[i] << std::endl;
	//}
	//std::cout << "cdfs_sizes: " << cdfs_sizes.size()<< std::endl;
	//std::cout << "offsets: " << offsets.size()<< std::endl;

	BufferedRansEncoder buffered_rans_enc;
    buffered_rans_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets);
  	std::string result = buffered_rans_enc.flush();
    //std::cout << "Output string:\n" << result << std::endl;

  	auto options = torch::TensorOptions().dtype(torch::kUInt8);
  	torch::Tensor output = torch::from_blob(result.data(), result.size(), options);

  	_state = 1;
	//std::cout << "RansEncoder::encode_with_indexes END" << std::endl;
    //std::cout << "Output Tensor:\n" << output << std::endl;
  	return output.clone();
}


RansDecoder::RansDecoder(int64_t init) : _state(init){}

std::vector<int64_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<int64_t> &indexes_i,
                                 const std::vector<std::vector<int64_t>> &cdfs_i,
                                 const std::vector<int64_t> &cdfs_sizes_i,
                                 const std::vector<int64_t> &offsets_i) {

	//std::cout << "RansDecoder::decode_with_indexes START" << std::endl;

	//transform inputs to original types
	std::vector<int32_t> indexes(indexes_i.size());
	std::vector<std::vector<int32_t>> cdfs(cdfs_i.size());
	std::vector<int32_t> cdfs_sizes(cdfs_sizes_i.size());
	std::vector<int32_t> offsets(offsets_i.size());
	std::transform(indexes_i.begin(), indexes_i.end(), indexes.begin(), to_int32);
	for(int i=0; i<cdfs_i.size(); i++ ){
		std::vector<int64_t> v = cdfs_i[i];
		cdfs[i] = std::vector<int32_t>(v.size());
		std::transform(v.begin(), v.end(), cdfs[i].begin(), to_int32);
	}
	std::transform(cdfs_sizes_i.begin(), cdfs_sizes_i.end(), cdfs_sizes.begin(), to_int32);
	std::transform(offsets_i.begin(), offsets_i.end(), offsets.begin(), to_int32);

	//std::cout << "encoded: " << encoded<< std::endl;
	//std::cout << "indexes: " << indexes_i << "| " << indexes<< std::endl;
	//std::cout << "cdfs: ";
	//for(int i=0; i<cdfs_i.size(); i++ ){
	//	std::cout << cdfs_i[i] << "| " << cdfs[i] << std::endl;
	//}
	//std::cout << "cdfs_sizes: " << cdfs_sizes_i << "| " << cdfs_sizes<< std::endl;
	//std::cout << "offsets: " << offsets_i << "| " << offsets<< std::endl;

  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }
  _state = 1;

  //transform output original types to required type
  std::vector<int64_t> output_o(output.size());
  std::transform(output.begin(), output.end(), output_o.begin(), to_int64);

  //std::cout << "RansDecoder::decode_with_indexes END" << std::endl;
  //std::cout << "output: " << output << "| " << output_o<< std::endl;

  return output_o;
}

void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
  assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int32_t>
RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

TORCH_LIBRARY(tiny_compressai_rans, m) {
  m.class_<RansEncoder>("RansEncoder")
     .def(torch::init<int64_t>())
	 .def("get_state", [](const c10::intrusive_ptr<RansEncoder>& self) {
	      return self->_state; })
     .def("encode_with_indexes", &RansEncoder::encode_with_indexes)
	 .def_pickle(
	           [](const c10::intrusive_ptr<RansEncoder>& self) -> int64_t {
	             return self->_state;
	           },
	           [](int64_t state) -> c10::intrusive_ptr<RansEncoder> {
	             return c10::make_intrusive<RansEncoder>(std::move(state));
	           });
  ;

  m.class_<RansDecoder>("RansDecoder")
     .def(torch::init<int64_t>())
     .def("decode_with_indexes", &RansDecoder::decode_with_indexes)
	 .def_pickle(
	           [](const c10::intrusive_ptr<RansDecoder>& self) -> int64_t {
	             return self->_state;
	           },
	           [](int64_t state) -> c10::intrusive_ptr<RansDecoder> {
	             return c10::make_intrusive<RansDecoder>(std::move(state));
	           });

  ;
}


//PYBIND11_MODULE(ans, m) {
//  m.attr("__name__") = "tiny_compressai.ans";
//  m.doc() = "range Asymmetric Numeral System python bindings";
//  py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder")
//      .def(py::init<>())
//      .def("encode_with_indexes", &BufferedRansEncoder::encode_with_indexes)
//      .def("flush", &BufferedRansEncoder::flush);
//
//  py::class_<RansEncoder>(m, "RansEncoder")
//      .def(py::init<>())
//      .def("encode_with_indexes", &RansEncoder::encode_with_indexes);
//
//  py::class_<RansDecoder>(m, "RansDecoder")
//      .def(py::init<>())
//      .def("set_stream", &RansDecoder::set_stream)
//      .def("decode_stream", &RansDecoder::decode_stream)
//      .def("decode_with_indexes", &RansDecoder::decode_with_indexes,
//           "Decode a string to a list of symbols");
//}
