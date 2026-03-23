
## Inspiration 
https://www.promptfoo.dev/docs/configuration/testing-llm-chains/

## setup 

```bash 
##
uv init 
 
## 
uv add langchain langchain-core langchain-community langchain-openai openai

``` 



## promptfooconfig.yaml 

```yaml
  prompts:                                                                  
    - "{{question}}"                                                        
                                                                            
  providers:                                                                
    - exec: python llm_chain.py                                             
                                                                            
  tests:                                                                    
    - vars:                                                                 
        question: What is the capital of France?                            
    - vars:                                                                 
        question: Explain quantum entanglement in one sentence.   

```