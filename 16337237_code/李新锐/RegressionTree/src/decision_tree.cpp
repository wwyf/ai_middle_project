//
// Created by 李新锐 on 03/10/2018.
//

#include "decision_tree.h"
#include "random_lib.h"

/*
* 对应伪代码中的train函数
* Node为节点类型
* judge_func是一个函数参数，它的不同决定了决策树是ID3/还是CART
*/
void DecisionTree::train_worker(DecisionTree::Node *node, const JudgeFunc_t& judgeFunc)
{
    //获取该节点全部数据集的Y值
    auto D_Y = node->D.row(-1);
    //计算众数
    auto mode = (double)D_Y.sum() >= (node->D.cols() / 2.0);
    //若A为空集
    if(node->A.empty())
    {
        node->isLeaf = true;
        node->Y = mode;
        return;
    }
    //若D中全部样本属于同一类型
    if(all_equal(D_Y))
    {
        node->isLeaf = true;
        node->Y = D_Y[0];
        return;
    }
    //若所有样本各个属性值均相同
    bool flag = true;
    for(Index i = 0; i < node->D.rows() - 1; ++i)
    {
        if(!all_equal(node->D.row(i)))
        {
            flag = false;
            break;
        }
    }
    if(flag)
    {
        node->isLeaf = true;
        node->Y = mode;
        return;
    }
    //对于每一种特征计算信息增益或基尼系数的减少值
    Vec<double> info_gain;
    for(auto i: node->A)
        info_gain.push_back(judgeFunc(node->D, i, featureValues[i]));
    //a_star是信息增益最大的一组特征
    auto a_star = node->A[max_element(info_gain) - std::begin(info_gain)];
    node->C = a_star;
    //依据该组特征值划分数据集
    Vec<Vec<Index>> S(featureValues[a_star]);
    for(int i = 0; i < node->D.cols(); ++i)
    {
        S[node->D(a_star, i)].push_back(i);
    }
    //在特征空间中移除a_star
    node->A.erase(ranges::v3::find(node->A, a_star));
    //对于a_star的每个取值
    for(int i = 0; i < featureValues[a_star]; ++i)
    {
        // 取对应属性值相同的子训练集
        auto n = new Node(matrix_view<int>(node->D, S[i]), node->A);
        // 若子训练集为空，则将这个子节点标为叶节点，无需递归下去
        if(S[i].empty())
        {
            n->isLeaf = true;
            n->Y = mode;
            node->child.push_back(n);
        }
        else
        {
            //递归训练并添加到子节点集中
            train_worker(n, judgeFunc);
            node->child.push_back(n);
        }
    }
}

std::string DecisionTree::print_worker(DecisionTree::Node *node, int n, int cn, std::map<int, std::vector<std::string> > &mp, string trace)
{
    std::stringstream ss;
    string node_name = trace + "F" +  to_string(node->C) + "C" + to_string(cn);
    for(auto& i : node_name) if(i == '-') i = '_';
    if(node->isLeaf)
        ss << node_name << "[label=\"" << mp[node->C][node->Y] << "\"];\n";
    else
        ss << node_name << "[label=\"" << mp[node->C][0] << "\"];\n";
    for(size_t c = 0; c < node->child.size(); ++c)
    {
        if(node->child[c])
        {
            string child_name = node_name + "F" + to_string(node->child[c]->C) + "C" + to_string(c);;
            for(auto& i : child_name) if(i == '-') i = '_';
            ss << node_name << "->" << child_name << "[label=\"" << mp[node->C][c + 1] << "\"];\n";
            ss << print_worker(node->child[c], n+1, c, mp, node_name);
        }
    }
    return ss.str();
}


int DecisionTree::predict(const Eigen::Matrix<int, Eigen::Dynamic, 1>& X, predictOne)
{
    auto ptr = root;
    while(!ptr->isLeaf)
    {
        int c = X(ptr->C, 0);
        ptr = ptr->child[c];
    }
    return ptr->Y;
}

Vec<int> DecisionTree::predict(const matrix_view<int>& X)
{
    Vec<int> ret;
    if(X.empty()) return ret;
    for(Eigen::Index i = 0; i < X.cols(); ++i)
    {
        ret.push_back(predict(X.col(i), predictOne{}));
    }
    return ret;
}

double DecisionTree::vaild(const matrix_view<int>& X)
{
    auto ret = predict(X);
    auto n = X.cols();
    double correct = 0;
    for(Eigen::Index i = 0; i < n; ++i)
    {
        if(X(-1, i) == ret[i])
            correct++;
    }
    return correct / n;
}

DecisionTree::DecisionTree(const matrix_view<int> &_trainSet, const Vec<int> &_featureValues)
    :trainSet(_trainSet),
      featureCount(_featureValues.size()),
      featureValues(_featureValues),
      root(new Node(
               matrix_view<int>(trainSet),
               range(0, featureCount)))
{}

void DecisionTree::train(const JudgeFunc_t &judgeFunc)
{
    train_worker(root, judgeFunc);
}

string DecisionTree::print(std::map<int, std::vector<std::string> > &mp)
{
    std::stringstream s;
    s << "digraph {\n";
    s << print_worker(root, 0, 0, mp, "");
    s << "}\n";
    return s.str();
}

//PEP剪枝函数，返回值是当前节点下的叶子节点个数
int DecisionTree::prune_worker(Node* node)
{
    //若当前节点是叶子节点，直接返回1
    if(node->isLeaf)
        return 1;
    else
    {
        //统计叶子节点个数
        int num_leaf = 0;
        for(auto& i: node->child)
        {
            num_leaf += prune_worker(i);
        }
        //统计当前训练集上的错误个数error
        auto ret = predict(node->D);
        auto num_X = node->D.cols();
        double error = 0;
        for(Eigen::Index i = 0; i < num_X; ++i)
        {
            if(node->D(-1, i) != ret[i])
                error++;
        }
        //计算错误率ec
        constexpr double punish = 0.5;
        auto ec = (error + punish * num_leaf) / num_X;
        //假设样本为伯努利分布，计算标准差SD
        auto SD = std::sqrt(num_X * ec * (1-ec));
        //计算剪枝后的错误个数new_error
        auto D_Y = node->D.row(-1);
        auto mode = (double)D_Y.sum() >= (node->D.cols() / 2.0);
        double new_error = 0;
        for(Eigen::Index i = 0; i < num_X; ++i)
        {
            if(node->D(-1, i) != mode)
                new_error++;
        }
        //若剪枝后错误率更低，则剪枝
        if(error + SD > new_error + punish)
        {
            //将当前节点设为叶子节点，结果设为众数
            node->isLeaf = true;
            node->C = -1;
            node->Y = mode;
            //删除子节点
            for(auto &i: node->child)
            {
                delete(i);
            }
            node->child.clear();
            //返回叶子节点数为1
            return 1;
        }
        //否则不剪枝
        else
            return num_leaf;
    }
}

void DecisionTree::prune()
{
    prune_worker(root);
}

double JudgeFunc::H(double p)
{
    auto p1 = p;
    auto p2 = 1-p;
    auto ret = -(p1 * std::log(p1)) - (p2 * std::log(p2));
    if(std::isnan(ret))
        cout << "p1 = " << p1 << " p2 = " << p2 << endl;
    return -(p1 * std::log(p1)) - (p2 * std::log(p2));
}

double JudgeFunc::JudgeBaseFunc(const matrix_view<int>& D, int feature, int featureVal, const EntropyFunc_t& EntropyFunc, bool splitInfoFlag)
{
    auto D_Y = D.row(-1);
    auto p = D_Y.sum() * 1.0 / D_Y.size();
    auto H_D = H(p);
    Vec<pair<int, int>> S(featureVal);
    int n = D.cols();
    for(int i = 0; i < n; ++i)
    {
        S[D(feature, i)].first++;
        S[D(feature, i)].second += D(-1, i);
    }
    //TODO
    double H_D_A = 0;
    double SplitInfo = 0;
    for(auto& p: S)
    {
        if(p.first == 0 || p.second == 0|| p.first == p.second)
            continue;
        H_D_A += (p.first * 1.0 / n) * EntropyFunc(p.second * 1.0 / p.first);
        SplitInfo += (-(p.first*1.0/n) * std::log(p.first*1.0/n));
    }
    auto Gain_D =  H_D - H_D_A;
    if(splitInfoFlag)
        return Gain_D / SplitInfo;
    else
        return Gain_D;
}

double JudgeFunc::ID3(const matrix_view<int> &D, int feature, int featureVal)
{
    return JudgeBaseFunc(D, feature, featureVal, H, false);
}

double JudgeFunc::C45(const matrix_view<int> &D, int feature, int featureVal)
{
    return JudgeBaseFunc(D, feature, featureVal, H, true);
}

double JudgeFunc::gini(double p)
{
    return 2 * p * (1 - p);
}

double JudgeFunc::CART(const matrix_view<int> &D, int feature, int featureVal)
{
    return JudgeBaseFunc(D, feature, featureVal, gini, false);
}

Matrix vectorizeData(const FileData_t & fileData, vector<map<string, int>>& mp)
{
    if(fileData.empty())
        return Matrix{};
    Matrix ret(fileData[0].size(), fileData.size());
    try{
        for(Eigen::Index j = 0; j < ret.cols(); ++j)
        {
            for(Eigen::Index i = 0; i < ret.rows(); ++i)
            {
                if(mp[i].count(fileData[j][i]) == 0)
                    throw(std::runtime_error("Can not translate " + fileData[j][i]));
                ret(i, j) = mp[i][fileData[j][i]];
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        cout << e.what() << endl;
    }
    return ret;
}

matrix_view<int> BaggingTree::sample_D(const matrix_view<int>& trainSet)
{
    std::vector<Eigen::Index> v;
    std::set<Eigen::Index> s;
    auto n = trainSet.cols();
    for(Eigen::Index i = 0; i < n; ++i)
    {
        s.insert(RandLib::uniform_rand(0, n - 1));
    }
    for(auto i : s)
    {
        v.push_back(i);
    }
    return matrix_view<int>(trainSet, v);
}

Vec<Eigen::Index> BaggingTree::sample_A(size_t n, size_t k)
{
    Vec<Eigen::Index> ret;
    if(k >= n)
    {
        return range(Eigen::Index(0), Eigen::Index(n));
    }
    std::set<Eigen::Index> sel;
    while(sel.size() < k)
    {
        int s = RandLib::uniform_rand(0, n - 1);
        sel.insert(s);
    }

    for(auto i : sel)
    {
        ret.push_back(i);
    }
    std::sort(ret.begin(), ret.end());

    return ret;
}

BaggingTree::BaggingTree(const matrix_view<int>& _trainSet, const Vec<int>& _featureValues)
    :trainSet(_trainSet), featureValues(_featureValues) {}

void BaggingTree::train(const JudgeFunc_t& judgeFunc, int M, int k)
{
    for(int i = 0; i < M; ++i)
    {
        matrix_view<int> sub_D = sample_D(trainSet);
        Vec<Eigen::Index> sub_A_idx = sample_A(featureValues.size(), k);
        sub_A_idx.push_back(-1);
        forests_A.push_back(sub_A_idx);
        // for(auto i : sub_A_idx) cout << i << ',';
        // cout << endl;
        sub_D.select_row(sub_A_idx);
        // cout << sub_D << endl;
        Vec<int> sub_A;
        for(int i = 0; i < k; ++i)
        {           
             sub_A.push_back(featureValues[sub_A_idx[i]]);
            //  cout << sub_A[i] << ',';
        }
        // cout << endl;
        DecisionTree* tree = new DecisionTree(sub_D, sub_A);
        forests.push_back(tree);
        tree->train(judgeFunc);
    }
}

Vec<int> BaggingTree::predict(const matrix_view<int>& X)
{
    Vec<double> rets(X.cols(), 0.0);
    Vec<int> ret(X.cols(), 0);
    if(X.empty()) return ret;
    for(size_t i = 0; i < forests.size(); ++i)
    {
        auto X_ = X;
        X_.select_row(forests_A[i]);
        auto oneret = forests[i]->predict(X_);
        for(size_t i = 0; i < oneret.size(); ++i)
        {
            rets[i] += oneret[i];
        }
    }
    for(size_t i = 0; i < rets.size(); ++i)
    {
        if(rets[i] > (forests.size() / 2.0))
            ret[i] = 1;
    }
    return ret;
}

double BaggingTree::vaild(const matrix_view<int>& X)
{
    auto ret = predict(X);
    auto n = X.cols();
    double correct = 0;
    for(Eigen::Index i = 0; i < n; ++i)
    {
        if(X(-1, i) == ret[i])
            correct++;
    }
    return correct / n;
}

void BaggingTree::prune()
{
    for(auto ptr: forests)
    {
        ptr->prune();
    }
}