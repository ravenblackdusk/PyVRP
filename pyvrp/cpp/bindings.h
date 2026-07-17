#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Matrix.h"
#include "Measure.h"

#include <type_traits>

// Used to separate template parameters that otherwise do not work well with
// macros.
#define PYVRP_PARAM_SEP ,

namespace pybind11::detail
{
// Type caster for matrices of integral measures.
template <pyvrp::MeasureType T, std::integral V>
struct type_caster<pyvrp::Matrix<pyvrp::Measure<T, V>>>
{
    PYBIND11_TYPE_CASTER(pyvrp::Matrix<pyvrp::Measure<T PYVRP_PARAM_SEP V>>,
                         _("numpy.ndarray[int]"));

    bool load(pybind11::handle src, bool convert)  // Python -> C++
    {
        if (!convert && !pybind11::array_t<V>::check_(src))
            return false;

        auto const style
            = pybind11::array::c_style | pybind11::array::forcecast;
        auto const buf = pybind11::array_t<V, style>::ensure(src);

        if (!buf || buf.ndim() != 2)
            throw pybind11::value_error("Expected 2D np.ndarray argument!");

        if (buf.size() == 0)  // then the default constructed object is already
            return true;      // OK, and we have nothing to do.

        std::vector<pyvrp::Measure<T, V>> data
            = {buf.data(), buf.data() + buf.size()};

        value = pyvrp::Matrix<pyvrp::Measure<T, V>>(
            std::move(data), buf.shape(0), buf.shape(1));

        return true;
    }

    static pybind11::handle
    cast(pyvrp::Matrix<pyvrp::Measure<T, V>> const &src,  // C++ -> Python
         [[maybe_unused]] pybind11::return_value_policy policy,
         pybind11::handle parent)
    {
        auto constexpr elemSize = sizeof(V);

        pybind11::array_t<V> array
            = {{src.numRows(), src.numCols()},           // shape
               {elemSize * src.numCols(), elemSize},     // strides
               reinterpret_cast<V const *>(src.data()),  // data
               parent};                                  // base

        // This is not pretty, but it makes the matrix non-writeable on the
        // Python side. That's needed because src is const, and we should
        // preserve that to avoid issues.
        pybind11::detail::array_proxy(array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;

        return array.release();
    }
};

// Caster for integral-valued measures.
template <pyvrp::MeasureType T, std::integral V>
struct type_caster<pyvrp::Measure<T, V>>
{
    PYBIND11_TYPE_CASTER(pyvrp::Measure<T PYVRP_PARAM_SEP V>, _("int"));

    bool load(pybind11::handle src, bool convert)  // Python -> C++
    {
        static_assert(sizeof(long long) >= sizeof(V));

        if (!convert && !PyLong_Check(src.ptr()))  // accept only int if no
            return false;                          // conversion is allowed.

        PyObject *tmp = PyNumber_Long(src.ptr());  // any argument for which
        if (!tmp)                                  // int() succeeds.
            return false;

        auto const raw = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);

        // See https://docs.python.org/3/c-api/long.html#c.PyLong_AsLongLong:
        // -1 is returned on overflow, and OverflowError is set.
        if (raw == -1 && PyErr_Occurred())
            throw pybind11::error_already_set();

        value = pyvrp::Measure<T, V>(raw);
        return !PyErr_Occurred();
    }

    static pybind11::handle
    cast(pyvrp::Measure<T, V> const &src,  // C++ -> Python
         [[maybe_unused]] pybind11::return_value_policy policy,
         [[maybe_unused]] pybind11::handle parent)
    {
        return PyLong_FromLongLong(src.get());
    }
};

// Type caster for boolean-valued matrices (stored as uint8_t), used for the
// edge existence matrices.
template <> struct type_caster<pyvrp::Matrix<uint8_t>>
{
    PYBIND11_TYPE_CASTER(pyvrp::Matrix<uint8_t>, _("numpy.ndarray[bool]"));

    bool load(pybind11::handle src, bool convert)  // Python -> C++
    {
        if (!convert && !pybind11::array_t<uint8_t>::check_(src))
            return false;

        auto const style
            = pybind11::array::c_style | pybind11::array::forcecast;
        auto const buf = pybind11::array_t<uint8_t, style>::ensure(src);

        if (!buf || buf.ndim() != 2)
            throw pybind11::value_error("Expected 2D np.ndarray argument!");

        if (buf.size() == 0)  // then the default constructed object is already
            return true;      // OK, and we have nothing to do.

        std::vector<uint8_t> data = {buf.data(), buf.data() + buf.size()};
        value = pyvrp::Matrix<uint8_t>(
            std::move(data), buf.shape(0), buf.shape(1));

        return true;
    }

    static pybind11::handle
    cast(pyvrp::Matrix<uint8_t> const &src,  // C++ -> Python
         [[maybe_unused]] pybind11::return_value_policy policy,
         pybind11::handle parent)
    {
        pybind11::array_t<bool> array
            = {{src.numRows(), src.numCols()},              // shape
               {src.numCols(), size_t(1)},                  // strides
               reinterpret_cast<bool const *>(src.data()),  // data
               parent};                                     // base

        pybind11::detail::array_proxy(array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;

        return array.release();
    }
};

// Caster for standalone Cost type. Python sees a pyvrp.Cost.Cost object that
// behaves like an int for arithmetic and equality with plain numbers, but
// supports lexicographic comparison with other Cost objects (msr first, then
// monetary cost). Construction from Python accepts an int, a Cost object, or
// a (missingSoftRequired, cost) tuple.
template <> struct type_caster<pyvrp::Cost>
{
    PYBIND11_TYPE_CASTER(pyvrp::Cost, _("Cost"));

    bool load(pybind11::handle src, bool convert)  // Python -> C++
    {
        // Accept a pyvrp.Cost.Cost object.
        if (pybind11::hasattr(src, "_msr") && pybind11::hasattr(src, "_cost"))
        {
            auto msr = src.attr("_msr").cast<int32_t>();
            auto cost = src.attr("_cost").cast<int64_t>();
            value = pyvrp::Cost(msr, cost);
            return true;
        }

        // Accept a (msr, cost) tuple.
        if (PyTuple_Check(src.ptr()))
        {
            auto tup = pybind11::reinterpret_borrow<pybind11::tuple>(src);
            if (tup.size() != 2)
                return false;

            auto msr = tup[0].cast<int32_t>();
            auto cost = tup[1].cast<int64_t>();
            value = pyvrp::Cost(msr, cost);
            return true;
        }

        // Accept a plain int.
        if (!convert && !PyLong_Check(src.ptr()))
            return false;

        PyObject *tmp = PyNumber_Long(src.ptr());
        if (!tmp)
            return false;

        auto const raw = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);

        if (raw == -1 && PyErr_Occurred())
            throw pybind11::error_already_set();

        value = pyvrp::Cost(raw);
        return !PyErr_Occurred();
    }

    static pybind11::handle
    cast(pyvrp::Cost const &src,  // C++ -> Python
         [[maybe_unused]] pybind11::return_value_policy policy,
         [[maybe_unused]] pybind11::handle parent)
    {
        auto cost_mod = pybind11::module_::import("pyvrp.Cost");
        auto cost_cls = cost_mod.attr("Cost");
        return cost_cls(src.missingSoftRequired(), src.get()).release();
    }
};

// Caster for floating point measures.
template <pyvrp::MeasureType T, std::floating_point V>
struct type_caster<pyvrp::Measure<T, V>>
{
    PYBIND11_TYPE_CASTER(pyvrp::Measure<T PYVRP_PARAM_SEP V>, _("float"));

    bool load(pybind11::handle src, bool convert)  // Python -> C++
    {
        if (!convert && !PyFloat_Check(src.ptr()))  // accept only float if no
            return false;                           // conversion is allowed.

        PyObject *tmp = PyNumber_Float(src.ptr());  // any argument for which
        if (!tmp)                                   // float() succeeds.
            return false;

        auto const raw = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);

        // See https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble:
        // on error, returns -1.0 and sets PyErr_Occurred.
        if (raw == -1.0 && PyErr_Occurred())
            throw pybind11::error_already_set();

        value = pyvrp::Measure<T, V>(raw);
        return !PyErr_Occurred();
    }

    static pybind11::handle
    cast(pyvrp::Measure<T, V> const &src,  // C++ -> Python
         [[maybe_unused]] pybind11::return_value_policy policy,
         [[maybe_unused]] pybind11::handle parent)
    {
        return PyFloat_FromDouble(src.get());
    }
};
}  // namespace pybind11::detail
