#include <vector>
#include <cassert>
#include <iterator>
#include <new>
#include <type_traits>

template <typename T>
class VectorLinkedList
{
private:
    struct Node
    {
        alignas(T) unsigned char storage[sizeof(T)];
        int next = -1;
        int prev = -1;
        bool alive = false;

        T *ptr() { return reinterpret_cast<T *>(storage); }
        const T *ptr() const { return reinterpret_cast<const T *>(storage); }
    };

    std::vector<Node> nodes;
    std::vector<int> free_list;

    int head = -1;
    int tail = -1;
    size_t sz = 0;

private:
    template <typename... Args>
    int alloc_node(Args &&...args)
    {
        int id;

        if (!free_list.empty())
        {
            id = free_list.back();
            free_list.pop_back();
        }
        else
        {
            id = nodes.size();
            nodes.emplace_back();
        }

        Node &n = nodes[id];
        new (n.storage) T(std::forward<Args>(args)...); // placement new
        n.next = n.prev = -1;
        n.alive = true;

        return id;
    }

    void destroy_node(int id)
    {
        Node &n = nodes[id];
        if (n.alive)
        {
            n.ptr()->~T();
            n.alive = false;
        }
    }

public:
    // ========================
    // iterator
    // ========================
    class iterator
    {
        int cur;
        VectorLinkedList *list;

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using reference = T &;
        using pointer = T *;

        iterator(int c, VectorLinkedList *l)
        : cur(c)
        , list(l)
        {
        }

        reference operator*() { return *list->nodes[cur].ptr(); }
        pointer operator->() { return list->nodes[cur].ptr(); }

        iterator &operator++()
        {
            cur = list->nodes[cur].next;
            return *this;
        }

        iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator &operator--()
        {
            if (cur == -1)
            {
                cur = list->tail;
            }
            else
            {
                cur = list->nodes[cur].prev;
            }
            return *this;
        }

        bool operator==(const iterator &other) const { return cur == other.cur; }
        bool operator!=(const iterator &other) const { return cur != other.cur; }

        int index() const { return cur; }
    };

    // ========================
    // 基本接口
    // ========================

    size_t size() const { return sz; }
    bool empty() const { return sz == 0; }

    void clear()
    {
        for (int i = 0; i < (int)nodes.size(); ++i)
        {
            destroy_node(i);
        }

        nodes.clear();
        free_list.clear();
        head = tail = -1;
        sz = 0;
    }

    ~VectorLinkedList() { clear(); }

    // ========================
    // 访问（物理索引）
    // ========================

    T &operator[](int idx)
    {
        assert(nodes[idx].alive);
        return *nodes[idx].ptr();
    }

    const T &operator[](int idx) const
    {
        assert(nodes[idx].alive);
        return *nodes[idx].ptr();
    }

    // ========================
    // 插入
    // ========================

    template <typename... Args>
    int push_back(Args &&...args)
    {
        int id = alloc_node(std::forward<Args>(args)...);

        if (tail == -1)
        {
            head = tail = id;
        }
        else
        {
            nodes[tail].next = id;
            nodes[id].prev = tail;
            tail = id;
        }

        ++sz;
        return id;
    }

    template <typename... Args>
    int push_front(Args &&...args)
    {
        int id = alloc_node(std::forward<Args>(args)...);

        if (head == -1)
        {
            head = tail = id;
        }
        else
        {
            nodes[id].next = head;
            nodes[head].prev = id;
            head = id;
        }

        ++sz;
        return id;
    }

    template <typename... Args>
    int insert_after(int pos, Args &&...args)
    {
        assert(pos >= 0 && pos < (int)nodes.size());
        assert(nodes[pos].alive);

        int id = alloc_node(std::forward<Args>(args)...);

        int nxt = nodes[pos].next;

        nodes[pos].next = id;
        nodes[id].prev = pos;
        nodes[id].next = nxt;

        if (nxt != -1)
        {
            nodes[nxt].prev = id;
        }
        else
        {
            tail = id;
        }

        ++sz;
        return id;
    }

    template <typename... Args>
    int insert_before(int pos, Args &&...args)
    {
        assert(pos >= 0 && pos < (int)nodes.size());
        assert(nodes[pos].alive);

        if (pos == head)
        {
            return push_front(std::forward<Args>(args)...);
        }

        return insert_after(nodes[pos].prev, std::forward<Args>(args)...);
    }

    // ========================
    // 删除
    // ========================

    void erase(int id)
    {
        assert(nodes[id].alive);

        int p = nodes[id].prev;
        int n = nodes[id].next;

        if (p != -1)
        {
            nodes[p].next = n;
        }
        else
        {
            head = n;
        }

        if (n != -1)
        {
            nodes[n].prev = p;
        }
        else
        {
            tail = p;
        }

        destroy_node(id);
        free_list.push_back(id);

        --sz;
    }

    // ========================
    // 迭代
    // ========================

    iterator begin() { return iterator(head, this); }
    iterator end() { return iterator(-1, this); }

    // ========================
    // 额外接口（vector-like）
    // ========================

    int front_index() const { return head; }
    int back_index() const { return tail; }

    T &front()
    {
        assert(head != -1);
        return *nodes[head].ptr();
    }

    T &back()
    {
        assert(tail != -1);
        return *nodes[tail].ptr();
    }

    int prev(int idx) const
    {
        assert(idx >= 0 && idx < (int)nodes.size());
        assert(nodes[idx].alive);
        return nodes[idx].prev;
    }
    int next(int idx) const
    {
        assert(idx >= 0 && idx < (int)nodes.size());
        assert(nodes[idx].alive);
        return nodes[idx].next;
    }
};