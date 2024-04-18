from .nodes.nodes import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS

# Gelber Text
#print("🔶"+"\033[33mThis is an example in yellow.\033[0m")
# Orangefarbener Text
#print("🔶"+"\033[38;5;208mThis is an example in orange.\033[0m")
#print("🔶"+"\033[38;5;220mThis is an example in gold.\033[0m")

# Helles Gelb
text = "Chaosaiart: visit our Discord. https://chaosaiart.com/discord"
padding = (80 - len(text)) // 2  # Berechne den Abstand für die Mitte
#print("\n\n")
#print("🔶"+"\033[38;5;229mChaosaiart : visit your Discord. https://chaosaiart.com/discord\033[0m")
print("\n\n")
print(" " * padding +"🔶"+"\033[38;5;229m" + text + "\033[0m")
print("\n\n")
WEB_DIRECTORY = "./web"