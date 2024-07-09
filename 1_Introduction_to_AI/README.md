**Preface**

Generative models have become widely popular in recent years. If you’re reading this file, you’ve probably interacted with a generative model at some point. Maybe you’ve used ChatGPT to generate text, used style transfer in apps like Instagram, or seen the deepfake videos that have been making headlines. These are all examples of generative models in action!

**Introduction**
What exactly is generative modeling? The high-level idea is to provide data to a model to train it so afterward it can generate new data that looks similar to the training data. For example, if I train a model on a dataset of images of cats, I can then use that model to generate new images of cats that look like they could have come from the original dataset. This is a powerful idea, and it has a wide range of applications, from creating novel images and videos to generating text with a specific style.

**Generating First Image**
 
 In this chapter, we will start by generating our first image using a generative model.
 diffusers is a popular library that provides access to state-of-the-art diffusion models. It’s a powerful, simple toolbox that allows us to quickly load and train diffusion models!
 We’ll use Stable Diffusion version 1.5, a diffusion model capable of generating high-quality images! If you browse the model website, you can read the model card, an essential document for discoverability and reproducibility. There, you can read about the model, how it was trained, intended use cases, and more.
 Given we have a model (Stable Diffusion) and a tool to use the model (diffusers), we can now generate our first image! When we load models, we’ll need to send them to a specific hardware device, such as CPU (cpu), GPU (cuda or cuda:0), or Mac hardware called Metal (mps). The following code will frequently appear in the repo: it assigns a variable to cuda:0 if a GPU is available;otherwise, it will use a CPU.
 ```bash
 import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
 we’ll load Stable Diffusion 1.5. diffusers offers a convenient, high-level wrapper called StableDiffusionPipeline, which is ideal for this use case. Don’t worry about all the parameters for now - the highlights are:

- There are many models with the Stable Diffusion architecture, so we need to specify the one we want to use, [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), in this case.

- We need to specify the precision we’ll load the model with. Precision is something we’ll learn more about later. At a high level, generative models are composed of many parameters (millions or billions of them). Each parameter is a number learned during training, and we can store these parameters with different levels of precision (in other words, we can use more bits to store the model). A larger precision means more memory and computation but usually also means a better model. On the other hand, we can use a lower precision by setting torch_dtype=float16 and use less memory than the default float32

Play with the prompt and generate new images.
You will notice that the generations could improve.

**Generating Our first text**
Just as diffusers is a very convenient library for diffusion models, the popular transformers library is extremely useful for running transformers-based models and adapting to new use cases. It provides a standardized interface for a wide range of tasks, such as generating text, detecting objects in images, and transcribing an audio file into text.

The transformers library provides different layers of abstractions. For example, if you don’t care about all the internals, the easiest is to use pipeline, which abstracts all the processing required to get a prediction. We can instantiate a pipeline by calling the pipeline() function and specifying which task we want to solve, such as text-classification.

Similarly, we can switch the task to text generation (text-generation), with which we can generate new text based on an input prompt. By default, the pipeline will use the GPT-2 model.

Although GPT-2 is not a great model by today’s standards, it gives us an initial example of transformers’ generation capabilities while using a small model. The same concepts we learn about GPT-2 can be applied to models such as Llama or Mistral, some of the most powerful open-access models.

**Generating our first sound clip**

Generative models are not limited to images and text. Models can generate videos, short songs, synthetic spoken speech, protein proposals, and more!
we can limit ourselves to the now familiar transformers pipeline and use the small version of MusicGen, a model released by Meta to generate music conditioned on text.

**Ethical and social effect**
While generative models offer remarkable capabilities, their widespread adoption raises important considerations around ethics and societal impact. It’s important to keep them in mind as we explore the capabilities of generative models. Here are a few key areas to consider:

- Privacy and consent: The ability of generative models to generate realistic images and videos based on very little data poses significant challenges to privacy. For example, creating synthetic images from a small set of real images from an individual raises questions about using personal data without consent. It also increases the risk of creating deepfakes.

- Bias and fairness: Generative models are trained on large datasets that contain biases. These biases can be inherited and amplified by the generative models. For example, biased datasets used to train image generation models may generate stereotypical or discriminatory images.

- Regulation: Given the potential risks associated with generative models, there is a growing call for regulatory oversight and accountability mechanisms to ensure responsible development and development. 

**How Are Generative AI Models Created? Big Budgets and Open Source**
Several of the most impressive generative models we’ve seen in the past couple of years were created by influential research labs in big, private companies. OpenAI developed ChatGPT, DALL·E, and Sora; Google built Imagen, Bard, and Gemini; and Meta created Llama and Code Llama.

In some cases, code and model weights are released as well: these are usually called open-source releases because those are the essential artifacts necessary to run the model on your hardware. Frequently, however, they are kept hidden for strategic reasons.

Big models, even when hidden, serve as inspiration for the community, whose work yields fruits that serve the field as a whole.

This cycle can only work because some of the models are open-sourced and can be used by the community. Companies that release open-source models don’t do it for altruistic reasons but because they see economic value in this strategy. By providing code and models that are adopted by the community, they receive public scrutiny with bug fixes, new ideas, derived model architectures, or even new datasets that work well with the models released.

**So we can replicate Open Source AI Right?**
At this point, we’d like to clarify that model releases are rarely truly open-source. Unlike in the software world, source code is not enough to fully understand a machine learning system. Model weights are not enough either: they are just the final output of the model training process. Being able to exactly replicate an existing model would require the source code used to train the model (not just the modeling code or the inference code), the training regime and parameters, and, crucially, all the data used for training. None of these, and particularly the data, are usually released.
