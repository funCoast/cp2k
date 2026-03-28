#pragma once
#include <iterator>
#include <stddef.h>
#include <utility>
// 完整的Enumerate类，支持结构化绑定（C++17及以上）
template <typename Iterator>
class Enumerate
{
private:
    Iterator begin_;
    Iterator end_;
    size_t start_;

public:
    // 嵌套的迭代器类
    class IteratorWrapper
    {
    private:
        size_t index_;
        Iterator iter_;

    public:
        IteratorWrapper(size_t index, Iterator iter)
        : index_(index)
        , iter_(iter)
        {
        }

        // 解引用操作符，返回pair<索引, 引用>
        auto operator*() const { return std::pair<size_t, decltype(*iter_)>{index_, *iter_}; }

        // 前缀++
        IteratorWrapper &operator++()
        {
            ++index_;
            ++iter_;
            return *this;
        }

        // 后缀++
        IteratorWrapper operator++(int)
        {
            IteratorWrapper temp = *this;
            ++(*this);
            return temp;
        }

        // 比较操作符
        bool operator!=(const IteratorWrapper &other) const { return iter_ != other.iter_; }

        bool operator==(const IteratorWrapper &other) const { return iter_ == other.iter_; }
    };

    // Enumerate构造函数
    Enumerate(Iterator begin, Iterator end, size_t start = 0)
    : begin_(begin)
    , end_(end)
    , start_(start)
    {
    }

    // 获取开始迭代器
    IteratorWrapper begin() const { return IteratorWrapper(start_, begin_); }

    // 获取结束迭代器
    IteratorWrapper end() const { return IteratorWrapper(start_ + std::distance(begin_, end_), end_); }
};

// 辅助函数，类似于Python的enumerate
template <typename Container>
auto enumerate(Container &container, size_t start = 0)
{
    return Enumerate(container.begin(), container.end(), start);
}

// 常量版本
template <typename Container>
auto enumerate(const Container &container, size_t start = 0)
{
    return Enumerate(container.begin(), container.end(), start);
}