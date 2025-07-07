#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <type_traits>

namespace utec::algebra {

    template <typename T, size_t Rank>
    class Tensor {
    private:
        std::vector<T> data_;
        std::array<size_t, Rank> shape_;
        size_t total_size_ = 1;

        template <typename U, size_t R>
        friend Tensor<U, R> transpose_2d(const Tensor<U, R>&);

    public:
        Tensor() = default;

        template <typename... Dims>
        Tensor(Dims... dims) {
            if (sizeof...(Dims) != Rank)
                throw std::runtime_error("Dimensiones incorrectas");
            shape_ = {static_cast<size_t>(dims)...};
            for (size_t d : shape_) total_size_ *= d;
            data_.resize(total_size_);
        }

        std::array<size_t, Rank> shape() const { return shape_; }

        size_t size() const { return total_size_; }

        template <typename... Indices>
        T& operator()(Indices... indices) {
            static_assert(sizeof...(Indices) == Rank, "Número de índices incorrecto");
            return data_[flatten_index({static_cast<size_t>(indices)...})];
        }

        template <typename... Indices>
        const T& operator()(Indices... indices) const {
            static_assert(sizeof...(Indices) == Rank, "Número de índices incorrecto");
            return data_[flatten_index({static_cast<size_t>(indices)...})];
        }

        void fill(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        Tensor& operator=(std::initializer_list<T> values) {
            if (values.size() != total_size_)
                throw std::runtime_error("Cantidad de datos no coincide");
            std::copy(values.begin(), values.end(), data_.begin());
            return *this;
        }

        // Agregado: operator[] para Tensor 2D
        Tensor<T, 2> operator[](size_t i) const {
            static_assert(Rank == 2, "operator[] solo soportado en Tensor 2D");
            size_t cols = shape_[1];
            Tensor<T, 2> row(1, cols);
            for (size_t j = 0; j < cols; ++j)
                row(0, j) = (*this)(i, j);
            return row;
        }

        template <typename... NewDims>
        void reshape(NewDims... dims) {
            if (sizeof...(NewDims) != Rank)
                throw std::runtime_error("N° de dimensiones incorrecto");
            std::array<size_t, Rank> new_shape = {static_cast<size_t>(dims)...};
            size_t new_total = 1;
            for (auto d : new_shape) new_total *= d;
            if (new_total > data_.size())
                throw std::runtime_error("Nueva forma excede tamaño");
            shape_ = new_shape;
            total_size_ = new_total;
            data_.resize(total_size_);
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            print_tensor(os, t, 0, 0);
            return os;
        }

        static void print_tensor(std::ostream& os, const Tensor& t, size_t dim, size_t index, size_t indent = 0) {
            if (dim == Rank - 1) {
                os << std::string(indent, ' ') << "";
                for (size_t i = 0; i < t.shape_[dim]; ++i) {
                    os << t.data_[index + i];
                    if (i + 1 < t.shape_[dim]) os << " ";
                }
                os << "";
            } else {
                os << std::string(indent, ' ') << "{\n";
                size_t step = 1;
                for (size_t d = dim + 1; d < Rank; ++d)
                    step *= t.shape_[d];
                for (size_t i = 0; i < t.shape_[dim]; ++i) {
                    print_tensor(os, t, dim + 1, index + i * step, indent + 2);
                    if (i + 1 < t.shape_[dim]) os << "\n";
                }
                os << "\n" << std::string(indent, ' ') << "}";
            }
        }

        size_t flatten_index(const std::array<size_t, Rank>& indices) const {
            size_t idx = 0, multiplier = 1;
            for (size_t i = Rank; i-- > 0;) {
                idx += indices[i] * multiplier;
                multiplier *= shape_[i];
            }
            return idx;
        }

        auto begin()             { return data_.begin(); }
        auto end()               { return data_.end(); }
        auto begin() const       { return data_.begin(); }
        auto end()   const       { return data_.end(); }
        auto cbegin() const      { return data_.cbegin(); }
        auto cend()   const      { return data_.cend(); }

        Tensor operator+(T scalar) const {
            Tensor result = *this;
            for (T& v : result.data_) v += scalar;
            return result;
        }
        Tensor operator-(T scalar) const {
            Tensor result = *this;
            for (T& v : result.data_) v -= scalar;
            return result;
        }
        Tensor operator*(T scalar) const {
            Tensor result = *this;
            for (T& v : result.data_) v *= scalar;
            return result;
        }
        Tensor operator/(T scalar) const {
            Tensor result = *this;
            for (T& v : result.data_) v /= scalar;
            return result;
        }

        friend Tensor operator+(T scalar, const Tensor& t) { return t + scalar; }
        friend Tensor operator-(T scalar, const Tensor& t) {
            Tensor result = t;
            for (T& v : result.data_) v = scalar - v;
            return result;
        }
        friend Tensor operator*(T scalar, const Tensor& t) { return t * scalar; }
        friend Tensor operator/(T scalar, const Tensor& t) {
            Tensor result = t;
            for (T& v : result.data_) v = scalar / v;
            return result;
        }
    };

    template <typename T, size_t Rank, size_t... Is>
    Tensor<T, Rank> create_tensor_with_shape(const std::array<size_t, Rank>& shape, std::index_sequence<Is...>) {
        return Tensor<T, Rank>(shape[Is]...);
    }

    template <typename T, size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& t) {
        if constexpr (Rank < 2)
            throw std::runtime_error("Transposición requiere al menos 2D");

        auto shape = t.shape();
        auto new_shape = shape;
        std::swap(new_shape[Rank - 1], new_shape[Rank - 2]);

        Tensor<T, Rank> result = create_tensor_with_shape<T>(new_shape, std::make_index_sequence<Rank>{});

        std::array<size_t, Rank> idx;
        for (size_t i = 0; i < t.total_size_; ++i) {
            size_t tmp = i;
            for (size_t j = Rank; j-- > 0;) {
                idx[j] = tmp % shape[j];
                tmp /= shape[j];
            }
            std::swap(idx[Rank - 1], idx[Rank - 2]);
            result.data_[result.flatten_index(idx)] = t.data_[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T, 2> matrix_product(const Tensor<T, 2>& a, const Tensor<T, 2>& b) {
        auto [m, k1] = a.shape();
        auto [k2, n] = b.shape();
        if (k1 != k2)
            throw std::runtime_error("Dimensiones incompatibles para producto");
        Tensor<T, 2> result(m, n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < k1; ++k)
                    result(i, j) += a(i, k) * b(k, j);
        return result;
    }

}

template<typename T, size_t Rank>
using Tensor = utec::algebra::Tensor<T, Rank>;
