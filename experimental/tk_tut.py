import tkinter as tk

def show_entry_fields():
    print(e1.get(), e2.get())

root = tk.Tk()

tk.Label(root,
         text="db").grid(row=0, column=0)
tk.Label(root,
         text="server").grid(row=1, column=0)

e1 = tk.Entry(root)
e2 = tk.Entry(root)

e1.grid(row=0, column=1)
e2.grid(row=2, columns=1)

tk.Button(root,
          text="Quit",
          command=root.quit).grid(row=3,
                                  column=0,
                                  sticky=tk.W,
                                  pady=4)

tk.Button(root,
          text="Show",
          command=show_entry_fields).grid(row=3,
                                          column=1,
                                          sticky=tk.W,
                                          pady=4)

root.mainloop()
