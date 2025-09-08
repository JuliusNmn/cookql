/**
 * @name Find bad methods
 * @description Find methods named 'bad' which typically contain vulnerabilities in Juliet testcases
 * @kind problem
 * @problem.severity warning
 * @id java/find-bad-methods
 */

import java

from Method m
where m.getName() = "bad"
select m, "Vulnerability method 'bad()' found in " + m.getDeclaringType().getName()

