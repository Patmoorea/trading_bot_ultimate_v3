#!/bin/bash
echo "# Audit Rapport" > REPORT.md
date >> REPORT.md
echo "\n## Pylint" >> REPORT.md
tail -n 20 pylint_report.txt >> REPORT.md
echo "\n## Radon" >> REPORT.md
tail -n 10 radon_report.txt >> REPORT.md
#!/bin/bash
echo "# Audit Rapport" > REPORT.md
date >> REPORT.md
echo "\n## Pylint" >> REPORT.md
tail -n 20 pylint_report.txt >> REPORT.md
echo "\n## Radon" >> REPORT.md
tail -n 10 radon_report.txt >> REPORT.md
