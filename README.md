# ParaTrace on Amazon SageMaker AI

ParaTrace tells whether an English text was written by a person or by AI. On AWS Marketplace it is an Amazon SageMaker
AI model package that you deploy in your own AWS account, as a real-time endpoint or as batch transform jobs. Your
texts stay in your account: SageMaker AI runs the model without network access, and nothing is stored.

This repository holds the sample notebook and the sample data of the AWS Marketplace listing.

| File | What |
|---|---|
| [`paratrace-on-sagemaker.ipynb`](paratrace-on-sagemaker.ipynb) | Deploys ParaTrace on ml.g4dn.xlarge, checks texts in real time, runs a batch transform job, and deletes what it made |
| [`samples/request.json`](samples/request.json) → [`samples/response.json`](samples/response.json) | A real-time request and its answer |
| [`samples/input.jsonl`](samples/input.jsonl) → [`samples/output.jsonl`](samples/output.jsonl) | A batch transform job's input and output |

The human texts in the samples are in the public domain: Abraham Lincoln's Gettysburg Address (1863) and the opening
of Jane Austen's *Pride and Prejudice* (1813). The AI texts were written by an AI assistant for these samples.

## Before you start

1. Subscribe to ParaTrace on AWS Marketplace.
2. Check your SageMaker AI quotas in Service Quotas: *ml.g4dn.xlarge for endpoint usage* for real-time endpoints,
   *ml.g4dn.xlarge for transform job usage* for batch transform. New accounts often have 0; ask for at least 1.
3. Open the notebook in SageMaker AI Studio or on a notebook instance whose role may create SageMaker AI models,
   endpoints and transform jobs, and use SageMaker AI's default S3 bucket (for example with the managed policy
   AmazonSageMakerFullAccess).

## Deploying: choose the newer SageMaker AI image

ParaTrace runs on ml.g4dn.xlarge, with one NVIDIA T4 GPU. SageMaker AI's default image for ml.g4dn has NVIDIA driver
470, which is too old for ParaTrace, so deploy it on a newer one:

| Deployment | Setting | Value |
|---|---|---|
| Real-time endpoint | `InferenceAmiVersion` of the endpoint configuration's production variant | `al2023-ami-sagemaker-inference-gpu-4-1` |
| Batch transform job | `TransformAmiVersion` of the job's `TransformResources` | `al2-ami-sagemaker-batch-gpu-535` |

Without it, the endpoint or job doesn't start, and its log in Amazon CloudWatch names the setting. The SageMaker AI
console's deploy page has no field for it: deploy with the notebook, the AWS CLI or an SDK.

With the AWS CLI, where `MODEL_PACKAGE_ARN` is the model package of your subscription in your Region and `ROLE_ARN`
a role SageMaker AI can assume:

```bash
aws sagemaker create-model --model-name paratrace --execution-role-arn "$ROLE_ARN" \
  --primary-container ModelPackageName="$MODEL_PACKAGE_ARN" --enable-network-isolation

# a real-time endpoint
aws sagemaker create-endpoint-config --endpoint-config-name paratrace --production-variants \
  VariantName=all,ModelName=paratrace,InstanceType=ml.g4dn.xlarge,InitialInstanceCount=1,InferenceAmiVersion=al2023-ami-sagemaker-inference-gpu-4-1
aws sagemaker create-endpoint --endpoint-name paratrace --endpoint-config-name paratrace
aws sagemaker wait endpoint-in-service --endpoint-name paratrace
aws sagemaker-runtime invoke-endpoint --endpoint-name paratrace --content-type application/json \
  --body fileb://samples/request.json response.json

# a batch transform job: every file under input/ gets its answers in output/<file>.out
aws sagemaker create-transform-job --transform-job-name paratrace-batch --model-name paratrace \
  --transform-input 'DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/input/}},ContentType=application/jsonlines,SplitType=Line' \
  --transform-output 'S3OutputPath=s3://<bucket>/output/,Accept=application/jsonlines,AssembleWith=Line' \
  --transform-resources InstanceType=ml.g4dn.xlarge,InstanceCount=1,TransformAmiVersion=al2-ami-sagemaker-batch-gpu-535
```

Delete the endpoint when you no longer need it: it is billed for every hour it runs.

## Requests

`POST /invocations` (InvokeEndpoint, or each request of a batch transform job) takes three content types.

**`application/json`**, for real-time endpoints:

```json
{"texts": ["first text ...", "second text ..."], "operating_point": "strict", "mode": "detailed"}
```

| Field | |
|---|---|
| `text` | One text to check; it is named `text` in the answer |
| `texts` | Several texts, named `text_1`, `text_2`, ... in the answer |
| `operating_point` | `strict` (the default): about 1 in 100 human texts is flagged. `lenient`: about 5 in 100, and fewer AI texts are missed |
| `mode` | `detailed` (the default): every verdict with its windows and segments. `fast`: the verdicts only |

A request holds up to 10 texts of up to 200,000 characters each. A JSON string alone is taken as `text`, and a list of
strings as `texts`. A real-time request must be answered within 60 seconds, which on ml.g4dn.xlarge is enough for
about 150,000 words; send more as a batch transform.

**`application/jsonlines`**, for batch transform jobs (`SplitType` `Line`): one text per line, either as a JSON
string or as `{"text": "...", "document": "a name", "operating_point": "...", "mode": "..."}`. The answer has one line
per line, in the same order. A line that can't be checked, for example because it isn't JSON or its text is empty,
gets `{"document": ..., "error": ...}`, and the other lines are still checked.

**`text/plain`**: one text, answered as `application/json`.

## Answers

`application/json` answers `{"detector_version": "1.0.4", "documents": [...]}` with one verdict per text, in the
order sent; `application/jsonlines` answers one verdict per line, each with `detector_version`.

| Field | |
|---|---|
| `document` | The text's name |
| `verdict` | `ai`, `human` or `mixed`: `ai` when at least 80% of the text's characters are labelled AI, `human` when at least 90% are labelled human |
| `is_ai` | The verdict isn't `human` |
| `probability` | P(AI-generated), 0 to 1. It ranks texts; the verdict decides them |
| `confidence` | `low`, `medium` or `high`: how far the evidence lies from the threshold |
| `ai_fraction` | The share of the characters labelled AI |
| `operating_point`, `threshold` | The setting used and its P(AI) threshold |
| `n_words`, `n_windows` | Words in the text as sent, and the 512-token windows it was scored in |
| `warnings` | For example `short text: verdict unreliable` under 50 words |
| `windows` | `detailed` only: each window's `start`, `end` and `probability` |
| `segments` | `detailed` only: runs of sentences with one label: `start`, `end`, `label` (`ai` or `human`) and `probability` |

`start` and `end` count Unicode code points in the text as sent, with `end` exclusive, so `text[start:end]` in Python
is the passage. `probability`, `is_ai`, `verdict` and `ai_fraction` are null for a text that is empty once cleaned.

## Speed on ml.g4dn.xlarge

Measured on an NVIDIA T4, the GPU of ml.g4dn.xlarge, with detailed answers:

| | |
|---|---|
| One text of 331 words | 0.14 s |
| One text of 2,022 words | 0.73 s |
| 40 texts, 10,607 words, one request | 2.95 s |
| Throughput | about 3,600 words a second: some 26,000 documents of 500 words an hour |

An endpoint takes about 5 minutes to start. A batch transform job takes about 10 minutes for a small input, most of
it starting the instance. Requests that arrive together share the GPU's passes, so a busy endpoint answers more per
second than one that gets requests one at a time.

## Accuracy

Measured on texts that were held out from training:

* **Human texts:** at `strict`, 1.3% of 2,081 human texts from ten kinds of writing are flagged. The kinds are
  abstracts, paper sections, books, short fiction, blog posts, emails, speeches, recipes, patents and open
  textbooks. Of the UChicago benchmark's human texts, 1.9% are flagged.
* **AI texts:** 99.6% of 26,086 answers from more than 100 chat models are caught. Of the Epoch AI benchmark's AI
  texts, 4.4% are missed, all of them imitations of a human author's style.
* **Confidence:** on texts half human and half AI, a text is human 0.1% of the time when judged `ai` with `high`
  confidence, 10% at `medium` and 34% at `low`. When judged `human`, it is human 85%, 95% and 99.3% of the time at
  `low`, `medium` and `high`.

## Limitations

* **English only.** Verdicts are estimates, not proof of who wrote a text.
* **Short texts:** under 50 words, verdicts are unreliable; the answer warns.
* **Calibration:** the thresholds are calibrated on a broad sample of human writing, and other writing has other
  false-positive rates. Human persuasive essays are the weak spot: 18.4% of them are flagged at `strict`.
* **Humanized AI text:** text rewritten by "humanizer" tools is often missed. Only 36% of text rewritten by
  StealthGPT is caught at `strict`.
* **Coarse segments:** a window that is a fifth AI can already score AI. AI segments therefore spread into the human
  text next to them, and `ai_fraction` runs high. A human passage shorter than one window (about 400 words) next to
  AI text is usually labelled AI with it.

## Support

contact@paratrace.net: deploying the model package, the request and answer formats, and
interpreting verdicts.
