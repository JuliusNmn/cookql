/**
 * @name SQL Injection via Statement.addBatch()
 * @description Detects possible SQL Injection when untrusted data flows into
 *              Statement.addBatch(String) without sanitization.
 * @kind problem
 * @problem.severity error
 * @id java/sql-injection-addBatch-simple
 * @tags security
 *       external/cwe/cwe-089
 */

import java
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.TaintTracking

/** A sink for SQL injection via Statement.addBatch() */
class AddBatchSink extends DataFlow::Node {
  AddBatchSink() {
    exists(MethodCall call |
      call.getMethod().hasName("addBatch") and
      call.getMethod().getDeclaringType().getAnAncestor().hasQualifiedName("java.sql", "Statement") and
      this.asExpr() = call.getArgument(0)
    )
  }
}

/** A taint-tracking configuration for SQL injection via addBatch. */
module AddBatchSqlInjectionConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    // Using FlowSources for broader coverage, but specifically System.getenv()
    source instanceof RemoteFlowSource or
    exists(MethodCall call |
      call.getMethod().hasName("getenv") and
      call.getMethod().getDeclaringType().hasQualifiedName("java.lang", "System") and
      source.asExpr() = call
    )
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof AddBatchSink
  }
}

module AddBatchSqlInjectionFlow = TaintTracking::Global<AddBatchSqlInjectionConfig>;

from AddBatchSqlInjectionFlow::PathNode source, AddBatchSqlInjectionFlow::PathNode sink
where AddBatchSqlInjectionFlow::flowPath(source, sink)
select sink.getNode(),
  "Possible SQL injection: data from " + source.getNode().toString() + 
  " reaches Statement.addBatch() without sanitization."
