.PHONY: push pushb pusht pushr pushf pushn

push:
	git add .
	git commit -m "$(message)"
	git push

pusht: message=typo fix
pusht: push

pushb: message=bug fix
pushb: push

pushf: message=feature enhancement
pushf: push

pushr: message=refactoring
pushr: push

pushn: message=update notes
pushn: push