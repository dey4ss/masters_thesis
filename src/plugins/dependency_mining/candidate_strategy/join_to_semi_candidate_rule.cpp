#include "join_to_semi_candidate_rule.hpp"

#include "expression/abstract_predicate_expression.hpp"
#include "expression/expression_functional.hpp"
#include "expression/expression_utils.hpp"
#include "logical_query_plan/join_node.hpp"
#include "logical_query_plan/predicate_node.hpp"

namespace opossum {

JoinToSemiCandidateRule::JoinToSemiCandidateRule() : AbstractDependencyCandidateRule(LQPNodeType::Join) {}

std::vector<DependencyCandidate> JoinToSemiCandidateRule::apply_to_node(
    const std::shared_ptr<const AbstractLQPNode>& lqp_node, const size_t priority) const {
  const auto join_node = static_pointer_cast<const JoinNode>(lqp_node);
  if (join_node->join_mode != JoinMode::Inner) return {};

  const auto& predicates = join_node->join_predicates();
  if (predicates.size() != 1) return {};

  const auto& predicate = std::static_pointer_cast<BinaryPredicateExpression>(predicates[0]);
  if (!predicate) return {};
  std::vector<DependencyCandidate> candidates;
  for (const auto& expression : {predicate->left_operand(), predicate->right_operand()}) {
    if (expression->type != ExpressionType::LQPColumn) return {};
    const auto join_column_id = resolve_column_expression(expression);
    if (join_column_id == INVALID_TABLE_COLUMN_ID) continue;
    candidates.emplace_back(TableColumnIDs{join_column_id}, TableColumnIDs{}, DependencyType::Unique, priority);
  }

  return candidates;
}

}  // namespace opossum