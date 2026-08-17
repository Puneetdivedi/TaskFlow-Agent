Plan:
1. Search capabilities: Implement a keyword-based search for tasks.
2. Task list improvements: Add filtering by priority and due date range.
3. Report formatting: Enhance the automation run report format.
- Implementation Plan:
    - Modify src/interfaces/task_store.py: Add search, filter params.
    - Modify src/memory/task_store.py: Implement search/filter logic.
    - Modify src/tools/task_tools.py: Update TaskTool to support new search/filter options.
    - Modify src/agent/automation.py: Improve report formatting.
