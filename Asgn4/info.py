#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 4
#

class Info():
    def __init__(self, name="", id="", email=""):
        self.name = name
        self.id = id
        self.email = email

    def __repr__(self):
        return "; ".join([self.name, self.id, self.email])

    def __str__(self):
        return "; ".join([self.name, self.id, self.email])


info = Info(
    name="Lee Tsz Yan",
    id="1155110177",
    email="1155110177@link.cuhk.edu.hk"
)
