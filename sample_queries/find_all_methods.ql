/**
 * @name Find all methods
 * @description Find all method declarations in the codebase
 * @kind problem
 * @problem.severity info
 * @id java/find-all-methods
 */

import java

from Method m
select m, "Method: " + m.getName() + " in class " + m.getDeclaringType().getName()

