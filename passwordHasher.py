import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['your pw of choice']).generate()
print(hashed_passwords)