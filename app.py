# Imports
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool

from langchain_groq import ChatGroq

import os
from groq import Groq
from utils import get_serper_api_key, get_groq_api_key

groq_api_key = get_groq_api_key()
os.environ["SERPER_API_KEY"] = get_serper_api_key()

# setting LLM settings
llm = ChatGroq(
    temperature=0, model_name="llama-3.1-70b-versatile", api_key=groq_api_key
)

# Listing down all the tools required by Agents
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# incase you have a written JOB post, put them in the same folder and rename job post file to "fake_job.md"
# same with personal data. If you have written everything and don't want us the UI, you can add the data as "fake_resume.md"
# you can run the app using "python run app.py" 
# Followng tools will be able to read through the provided link.But do note to change the "inputs" on line 217
read_resume = FileReadTool(file_path="./fake_resume.md")
read_job = FileReadTool(file_path="./fake_job.txt")
semantic_search_resume = MDXSearchTool(mdx="./fake_resume.md")
semantic_search_job = MDXSearchTool(mdx="./fake_job.md")

def cv_builder(
    jobLink: str,
    cv_data: str,
    github: str = "",
    user_writeup: str = "",
):
    llm = ChatGroq(
        temperature=0, model_name="llama-3.1-70b-versatile", api_key=groq_api_key
    )
    # Agent 1: Researcher
    researcher = Agent(
        role="Maintenance Job Researcher",
        goal=(
            "Make sure to do amazing analysis on "
            "job posting to help job applicants"
            "if a link is given, use scrap tool on the link to learn about the job"
            "otherwise go through the provided text"
        ),
        tools=[search_tool, read_job, scrape_tool, semantic_search_job],
        verbose=True,
        backstory=(
            "As a Job Researcher, your prowess in "
            "navigating and extracting critical "
            "information from {read_job} is unmatched."
            "Your skills help pinpoint the necessary "
            "qualifications and skills sought "
            "by employers, forming the foundation for "
            "effective application tailoring."
        ),
        llm=llm,
    )

    # Agent 2: Profiler
    profiler = Agent(
        role="Personal Profiler for Engineers",
        goal=(
            "Do increditble research on job applicants "
            "to help them stand out in the job market"
        ),
        tools=[
            scrape_tool,
            search_tool,
            read_resume,
            semantic_search_resume,
        ],
        verbose=True,
        backstory=(
            "Equipped with analytical prowess, you dissect "
            "and synthesize information "
            "from diverse sources to craft comprehensive "
            "personal and professional profiles, laying the "
            "groundwork for personalized resume enhancements. Take care about tokens and their use."
        ),
        llm=llm,
    )

    # Agent 3: Resume Strategist
    resume_strategist = Agent(
        role="Resume Strategist for Engineers",
        goal=("Find all the best ways to make a " "resume stand out in the job market."),
        tools=[read_resume, semantic_search_resume, read_job, semantic_search_job],
        verbose=True,
        backstory=(
            "With a strategic mind and an eye for detail, you "
            "excel at refining resumes to highlight the most "
            "relevant skills and experiences, ensuring they "
            "resonate perfectly with the job's requirements."
            "you use all the tools available to you to craft one."
        ),
        llm=llm,
    )

    # Agent 4: Interview Preparer
    interview_preparer = Agent(
        role="Engineering Interview Preparer",
        goal=(
            "Create interview questions and talking points "
            "based on the resume and job requirements"
            "if a link is given, go to the linked address and use scrap tool to learn about the job"
            "otherwise go through the provided text"
        ),
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=True,
        backstory=(
            "Your role is crucial in anticipating the dynamics of "
            "interviews. With your ability to formulate key questions "
            "and talking points, you prepare candidates for success, "
            "ensuring they can confidently address all aspects of the "
            "job they are applying for."
        ),
        llm=llm,
    )

    # Task for Researcher Agent: Extract Job Requirements
    research_task = Task(
        description=(
            "Analyze the job posting URL provided ({read_job}) "
            "to extract key skills, experiences, and qualifications "
            "required. Use the tools to gather content and identify "
            "and categorize the requirements."
        ),
        expected_output=(
            "A structured list of job requirements, including necessary "
            "skills, qualifications, and experiences."
        ),
        agent=researcher,
        async_execution=True,
    )

    # Task for Profiler Agent: Compile Comprehensive Profile
    profile_task = Task(
        description=(
            "Compile a detailed personal and professional profile "
            # "using the GitHub ({github_url}) URLs,  "
            "and personal write-up ({personal_writeup}). Utilize tools to extract and "
            "synthesize information from these sources."
        ),
        expected_output=(
            "A comprehensive profile document that includes skills, "
            "project experiences, contributions, interests, and "
            "communication style."
        ),
        agent=profiler,
        async_execution=True,
    )

    # Task for Resume Strategist Agent: Align Resume with Job Requirements
    resume_strategy_task = Task(
        description=(
            "Using the profile and job requirements obtained from "
            "previous tasks, tailor the resume to highlight the most "
            "relevant areas. Employ tools to adjust and enhance the "
            "resume content. Make sure this is the best resume even but "
            "don't make up any information. Update every section, "
            "inlcuding the initial summary, work experience, skills, "
            "and education. All to better reflect the candidates "
            "abilities and how it matches the job posting."
        ),
        expected_output=(
            "An updated resume that effectively highlights the candidate's "
            "qualifications and experiences relevant to the job."
        ),
        output_file="tailored_resume.md",
        context=[research_task, profile_task],
        agent=resume_strategist,
    )

    # Task for Interview Preparer Agent: Develop Interview Materials
    interview_preparation_task = Task(
        description=(
            "Create a set of potential interview questions and talking "
            "points based on the tailored resume and job requirements. "
            "Utilize tools to generate relevant questions and discussion "
            "points. Make sure to use these question and talking points to "
            "help the candiadte highlight the main points of the resume "
            "and how it matches the job posting."
        ),
        expected_output=(
            "A document containing key questions and talking points "
            "that the candidate should prepare for the initial interview."
        ),
        output_file="interview_materials.md",
        context=[research_task, profile_task, resume_strategy_task],
        agent=interview_preparer,
    )

    # Crew

    job_application_crew1 = Crew(
        agents=[
            researcher,
            profiler,
            resume_strategist,
            # interview_preparer
        ],
        tasks=[
            research_task,
            profile_task,
            resume_strategy_task,
            #    interview_preparation_task
        ],
        verbose=True,
    )

    inputs = {
        "read_job": jobLink,
        "read_resume": cv_data,
        "github": github,
        "personal_writeup": user_writeup,
    }

    result = job_application_crew1.kickoff(inputs=inputs)
    return result

cv_builder(jobLink="", cv_data="", github="", user_writeup="")
# from IPython.display import Markdown, display

# display(Markdown("./tailored_resume.md"))

# display(Markdown("./interview_materials.md"))
