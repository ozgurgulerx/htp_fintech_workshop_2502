# 🤖 Introduction to AI Automation with AutoGen  

We now know autogen is a multi-agent framework from **Microsoft Research**for building automated AI workflows.
...but why do we need these "frameworks" in the first place?

AI automation for non-deterministic and dynamic tasks requires creating a "workflow" that can be executed with code, external API calls and data access on the go. Two points are crucial here...

Let's work backwards from the needs for complex AI Automation and see how AutoGen can help us...


![AI Automation Tech Stack](../images/agent_tech_stack.png)

## AI Automation Requirements 

1. **Task decomposition & planning** We need to diminish the task into smaller sub-tasks that can be executed in parallel (where we can to untangle dependencies and turn calls into "asynchrounous calls"). This is where we need "reasoning llm's" for their "executive function" capabilities much like in humans. \
    -**Reasoning LLM's for task decomposition & planning** with  best in class instruction following \
    -**Multi-agent communication support**  asynchrnous and event-driven multi-agent communication is key to this. 
    
2. **Tool use** via "function calling" we need to be able to call code and APIs on the fly consistently and correctly. \
    -**Structured Outputs** LLM structured outputs support is key for consistent & correct function calls \
    -**Consistent Calls** LLM function calling performance should be consistent and correct \
    -**Instruction Following** Instruction following determines prompt performance e.g. how well you can your intent into LLM actions with Prompt Engineering. It is the prompt engineering elasticit... \
    -**Capable & optimal Worker LLM's** We need optimal worker LLM's performance, latency & cost 

3. **Episodic Memory** For complex tasks we need to store state, fix problems on the go, backtrack where necessary and adopt a different plan if the execution is failing... 
    - **Memory Management** 
    - **Context Management**

4. **Observability** is key for "interpretable" AI Automation implementations for debugging / troubleshooting and for instilling trust in users. 

5. **Agent Evals**  Evals is mandatory for any serious AI Automation project ensure consistent and high performance of the system as a whole...

--
## AutoGen - The "Why"? 
[Burayi AutoGen'i tanidikca dolduracagiz!!!]