// Level 3: 容器适配器 —— stack, queue, priority_queue
// 涵盖：LIFO/FIFO 语义、自定义底层容器、priority_queue 堆操作与自定义比较器

#include <functional>
#include <iostream>
#include <queue>
#include <stack>
#include <string>
#include <vector>

// ============================================================
// 3.1 std::stack —— 后进先出（LIFO）
//   默认底层容器：std::deque（也可选 std::vector / std::list）
//   接口：push, pop, top, empty, size
// ============================================================
void demo_stack() {
    std::cout << "=== 3.1 std::stack ===\n";

    std::stack<int> st;

    // push：压栈
    st.push(10);
    st.push(20);
    st.push(30);
    std::cout << "top=" << st.top() << "  size=" << st.size() << "\n";

    // pop：弹出栈顶（不返回值，需先用 top() 读取）
    st.pop();
    std::cout << "pop 后 top=" << st.top() << "\n";

    // 安全弹出惯用法
    while (!st.empty()) {
        std::cout << st.top() << " ";
        st.pop();
    }
    std::cout << "\n";

    // 指定底层容器为 vector（更节省内存、cache 友好）
    std::stack<int, std::vector<int>> st_vec;
    st_vec.push(1);
    st_vec.push(2);
    std::cout << "vector底层 top=" << st_vec.top() << "\n";
}

// ============================================================
// 3.2 应用：用栈判断括号匹配
// ============================================================
bool is_balanced(const std::string& s) {
    std::stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.empty()) return false;
            char top = st.top(); st.pop();
            if ((c == ')' && top != '(') ||
                (c == ']' && top != '[') ||
                (c == '}' && top != '{'))
                return false;
        }
    }
    return st.empty();
}

void demo_stack_usage() {
    std::cout << "\n=== 3.1 应用：括号匹配 ===\n";
    std::cout << "\"({[]})\" -> " << (is_balanced("({[]})") ? "合法" : "非法") << "\n";
    std::cout << "\"({[})\"  -> " << (is_balanced("({[})")  ? "合法" : "非法") << "\n";
}

// ============================================================
// 3.3 std::queue —— 先进先出（FIFO）
//   默认底层容器：std::deque
//   接口：push, pop, front, back, empty, size
// ============================================================
void demo_queue() {
    std::cout << "\n=== 3.3 std::queue ===\n";

    std::queue<int> q;

    // push：入队（尾部）
    q.push(1);
    q.push(2);
    q.push(3);
    std::cout << "front=" << q.front() << "  back=" << q.back()
              << "  size=" << q.size() << "\n";

    // pop：出队（前端，不返回值）
    q.pop();
    std::cout << "pop 后 front=" << q.front() << "\n";

    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << "\n";
}

// ============================================================
// 3.4 应用：BFS 层序遍历（简化树节点）
// ============================================================
struct TreeNode {
    int val;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
    TreeNode(int v) : val(v) {}
};

void bfs(TreeNode* root) {
    if (!root) return;
    std::queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            TreeNode* node = q.front(); q.pop();
            std::cout << node->val << " ";
            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
        std::cout << "\n";
    }
}

void demo_queue_usage() {
    std::cout << "\n=== 3.3 应用：BFS 层序遍历 ===\n";
    //        1
    //       / \
    //      2   3
    //     / \
    //    4   5
    TreeNode* root = new TreeNode(1);
    root->left       = new TreeNode(2);
    root->right      = new TreeNode(3);
    root->left->left  = new TreeNode(4);
    root->left->right = new TreeNode(5);
    bfs(root);
    // 清理
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;
}

// ============================================================
// 3.5 std::priority_queue —— 堆（优先队列）
//   默认：最大堆（堆顶是最大值）
//   底层：std::vector + make_heap 算法
//   接口：push, pop, top, empty, size
// ============================================================
void demo_priority_queue() {
    std::cout << "\n=== 3.5 std::priority_queue ===\n";

    // 最大堆（默认）
    std::priority_queue<int> maxpq;
    for (int x : {3, 1, 4, 1, 5, 9, 2, 6})
        maxpq.push(x);

    std::cout << "最大堆，依次弹出: ";
    while (!maxpq.empty()) {
        std::cout << maxpq.top() << " ";
        maxpq.pop();
    }
    std::cout << "\n";

    // 最小堆：第三模板参数传 std::greater<T>
    std::priority_queue<int, std::vector<int>, std::greater<int>> minpq;
    for (int x : {3, 1, 4, 1, 5, 9, 2, 6})
        minpq.push(x);

    std::cout << "最小堆，依次弹出: ";
    while (!minpq.empty()) {
        std::cout << minpq.top() << " ";
        minpq.pop();
    }
    std::cout << "\n";

    // 自定义类型 + lambda 比较器
    struct Task {
        int priority;
        std::string name;
    };
    auto cmp = [](const Task& a, const Task& b) {
        return a.priority < b.priority;  // 大 priority 优先（最大堆）
    };
    std::priority_queue<Task, std::vector<Task>, decltype(cmp)> taskq(cmp);
    taskq.push({3, "低优先"});
    taskq.push({9, "高优先"});
    taskq.push({6, "中优先"});

    std::cout << "任务调度顺序: ";
    while (!taskq.empty()) {
        std::cout << taskq.top().name << "(" << taskq.top().priority << ") ";
        taskq.pop();
    }
    std::cout << "\n";
}

// ============================================================
// 3.6 应用：求前 K 个最大数（最小堆维护大小为 K 的窗口）
// ============================================================
void demo_topk() {
    std::cout << "\n=== 3.5 应用：Top-K 最大值 ===\n";

    std::vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    int k = 3;

    // 维护一个大小为 k 的最小堆
    // 堆顶是已见元素中第 k 大的值
    std::priority_queue<int, std::vector<int>, std::greater<int>> minpq;
    for (int x : nums) {
        minpq.push(x);
        if ((int)minpq.size() > k)
            minpq.pop();
    }

    std::cout << "前 " << k << " 个最大值: ";
    while (!minpq.empty()) {
        std::cout << minpq.top() << " ";
        minpq.pop();
    }
    std::cout << "\n";
}

int main() {
    demo_stack();
    demo_stack_usage();
    demo_queue();
    demo_queue_usage();
    demo_priority_queue();
    demo_topk();
    return 0;
}
