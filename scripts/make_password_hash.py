import getpass
import bcrypt

def main():
    print("AutoChecker Password Hash Generator")
    password = getpass.getpass("Enter password to hash: ")
    if not password:
        print("Password cannot be empty.")
        return

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    print(f"\nGenerated Bcrypt Hash:\n{hashed}")
    print("\nCopy this hash into your .env file as AUTH_PASSWORD_HASH.")

if __name__ == "__main__":
    main()
