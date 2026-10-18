from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("NI HAO!")
    info = """
    Tywin is a tall, slender, broad-shouldered man in his fifties. His arms are thin but muscled. Tywin has kept his head shaved ever since his golden hair[10] started going bald, but he grows out bushy golden side-whiskers. Tywin has pale green eyes flecked with gold.[4][11][12][13]

    Tywin is a calculating, intelligent, politically astute, ruthless, and controlling man.[3] He dedicates his life and efforts towards maintaining House Lannister's prestige and ensuring his family is respected, or at least feared.[14] Tywin is a proven battle commander[15] who leads from the rear,[16][17][18] and he has a powerful presence combined with an intimidating gaze.[19] He is harsh with betrayal.[20] According to his sister Genna, Tywin mistrusts laughter because of how his father, Lord Tytos, had been laughed at;[21] Grand Maester Pycelle was amazed that his late wife Joanna was able to make him laugh three different times.[3]

    Though ruthless, Tywin is an able and shrewd ruler who brought great prosperity during his tenure as Hand of the King to Aerys II Targaryen.[3] He is especially talented at the raising of funds, leading to a persistent jest that he must "shit gold."[11] Smallfolk also say that Tywin has a heart of old hard gold.[22] Tywin is more respected than loved in the westerlands,[23] and he is remembered in King's Landing for the Sack.[23]

    Tywin considers bastards to be treacherous in nature[24] and public association with whores to be shameful to House Lannister.[22][12]


    Tywin loves his elder children, Queen Cersei and Ser Jaime, but despises Tyrion. This is partly because Tyrion is a dwarf, but also Tywin blames his youngest son for causing his beloved wife Joanna's death during his birth, as well as for shaming the family name with his frequent whoring. Tywin often relies on his brother, Ser Kevan Lannister.[25][21] Ser Addam Marbrand is said to be Tywin's most daring commander.[26] Genna's husband, Ser Emmon Frey, is terrified of his brother-in-law.[27][21]

    Tywin's steel plate armor is enameled in deep crimson and highlighted with gold, and its rondels are golden sunbursts. Tywin's heavy cloth-of-gold greatcloak is clasped with gold miniature lionesses, and his greathelm is decorated with a roaring lion with ruby eyes.[16][5] He also owns a thick ermine cloak.[26] Tywin's longsword has a gold pommel.[11] His charger is a stallion[16] with its own gilded armor.[5] The destrier has a blanket with enameled crimson scales and gilded crinet and chamfron,[26] as well as crimson silk bardings decorated with the lion of the Lannisters.[5]
    """

    summary = """
    given the information {info} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["info"], template=summary
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0,model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"info": info})
    print(response.content)


if __name__ == "__main__":
    main()
